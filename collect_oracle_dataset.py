import multiprocessing
import os
from pathlib import Path
from typing import NoReturn

import cv2
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from language_table.environments import blocks, language_table
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards.block2absolutelocation import ABSOLUTE_LOCATIONS
from language_table.environments.oracles import push_oracle_rrt_slowdown
from tf_agents.environments import gym_wrapper
from tf_agents.trajectories import time_step as ts

FPS = 10
MAX_EPISODE_STEPS = 20 * FPS  # 20-second timeout
IMG_H, IMG_W = 360, 640

FEATURES = {
    "observation.images.rgb": {
        "dtype": "video",
        "shape": (IMG_H, IMG_W, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["x", "y"]},
    },
    "observation.effector_target_translation": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["x", "y"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["x", "y"]},
    },
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.done": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
}

NUM_EPISODES = 1000
NUM_WORKERS = min(NUM_EPISODES, max(1, os.cpu_count() - 1))

repo_id = "sidhraja/language_table_blocktoabsolute_oracle_sim"
root = Path("./language_table_blocktoabsolute_oracle_sim_1000")

block_list = list(blocks.FIXED_4_COMBINATION)
location_list = list(ABSOLUTE_LOCATIONS.keys())


def switch_task(env, raw_env, block, location):
    state = raw_env._compute_state(request_task_update=False)
    info = raw_env._reward_calculator.reset_to(
        state, block, location, raw_env._blocks_on_table
    )
    raw_env._set_task_info(info)
    instruction = language_table.LanguageTable.decode_instruction(raw_env._instruction)
    obs = raw_env._compute_observation()
    restart_ts = ts.restart(obs)
    env._current_time_step = restart_ts
    env._done = False
    return restart_ts


def collect_one_episode(ep_idx, env, raw_env, oracle):
    """Returns list of (frames, instruction) for each successful block/location pair, or [] if all timed out."""
    rng = np.random.default_rng(seed=ep_idx)
    shuffled_blocks = rng.choice(block_list, size=len(block_list), replace=False)
    shuffled_locations = rng.choice(location_list, size=len(block_list), replace=False)
    block_location_pairs = list(zip(shuffled_blocks, shuffled_locations))

    env.reset()

    results = []
    for block, location in block_location_pairs:
        current_ts = switch_task(env, raw_env, block, location)
        oracle.reset()
        instruction = language_table.LanguageTable.decode_instruction(raw_env._instruction)

        buffered_frames = []
        success = False
        for _ in range(MAX_EPISODE_STEPS):
            action = oracle.action(current_ts, ()).action
            frame_rgb = raw_env.render(mode="rgb_array")
            frame_rgb = cv2.resize(frame_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
            obs = current_ts.observation

            buffered_frames.append({
                "observation.images.rgb": frame_rgb,
                "observation.state": np.array(obs["effector_translation"], dtype=np.float32),
                "observation.effector_target_translation": np.array(obs["effector_target_translation"], dtype=np.float32),
                "action": np.array(action, dtype=np.float32),
                "next.reward": np.array([current_ts.reward if current_ts.reward is not None else 0.0], dtype=np.float32),
                "next.done": np.array([current_ts.is_last()], dtype=bool),
                "task": instruction,
            })

            current_ts = env.step(action)
            if current_ts.is_last():
                success = True
                break

        if success and len(buffered_frames) >= 2:
            results.append((buffered_frames, instruction))
        elif success:
            print(f"  skipping 1-frame sub-task (already at goal): {instruction}")
        else:
            print(f"  block/location pair timed out, skipping")

    return results


def collect_episodes_worker(worker_id, work_queue, result_queue):
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
        reward_factory=block2absolutelocation.BlockToAbsoluteLocationReward,
        control_frequency=10.0,
        render_text_in_image=False,
        seed=worker_id,
    )
    env = gym_wrapper.GymWrapper(env)
    if not hasattr(env, "get_control_frequency"):
        env.get_control_frequency = lambda: env._control_frequency

    oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
        env, use_ee_planner=True
    )
    raw_env = env.gym

    while True:
        ep_idx = work_queue.get()
        if ep_idx is None:
            break

        print(f"[worker-{worker_id}] collecting episode {ep_idx}")
        results = collect_one_episode(ep_idx, env, raw_env, oracle)
        # put one item per sub-task result, plus a sentinel marking end of this ep_idx
        result_queue.put((ep_idx, results))

    env.close()


def main():
    work_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue(maxsize=NUM_WORKERS)

    for ep_idx in range(NUM_EPISODES):
        work_queue.put(ep_idx)
    for _ in range(NUM_WORKERS):
        work_queue.put(None)

    workers = [
        multiprocessing.Process(
            target=collect_episodes_worker,
            args=(i, work_queue, result_queue),
            daemon=True,
        )
        for i in range(NUM_WORKERS)
    ]
    for w in workers:
        w.start()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=FPS,
        robot_type="xarm",
        features=FEATURES,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=2,
    )

    saved = 0
    skipped = 0
    with tqdm(total=NUM_EPISODES, desc="episodes saved", unit="ep") as pbar:
        for _ in range(NUM_EPISODES):
            ep_idx, results = result_queue.get(timeout=600)
            for frames, instruction in results:
                for frame_data in frames:
                    dataset.add_frame(frame_data)
                dataset.save_episode()
                saved += 1
                pbar.update(1)
                pbar.set_postfix(saved=saved, skipped=skipped)
            skipped += len(block_list) - len(results)

    for w in workers:
        w.join()

    dataset.finalize()
    print(f"Done. Saved {saved} episodes, skipped {skipped} timed-out sub-tasks.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
