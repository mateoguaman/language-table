"""
Add k zero-action padding frames to the end of every episode in a LeRobot dataset.

Usage:
    python pad_dataset_with_zero_actions.py \
        --src ./language_table_blocktoabsolute_oracle_sim_1000 \
        --dst ./language_table_blocktoabsolute_oracle_sim_1000_padded \
        --k 10
"""

import argparse
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--src_repo_id", default=None)
    return parser.parse_args()


def read_episode_frames(container, from_ts: float, length: int) -> list[np.ndarray]:
    """Seek to from_ts (seconds) and decode exactly length frames."""
    stream = container.streams.video[0]
    target_pts = int(from_ts / stream.time_base)
    container.seek(target_pts, stream=stream, backward=True)
    frames = []
    for frame in container.decode(stream):
        frame_ts = float(frame.pts * stream.time_base)
        if frame_ts < from_ts - 1e-4:
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= length:
            break
    return frames


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    src_repo_id = args.src_repo_id or src_root.name
    dst_repo_id = args.repo_id or dst_root.name

    src = LeRobotDataset(src_repo_id, root=src_root, video_backend="pyav")

    raw_features = {}
    for key, feat in src.meta.features.items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue
        raw_features[key] = {k: v for k, v in feat.items() if k != "info"}

    dst = LeRobotDataset.create(
        repo_id=dst_repo_id,
        root=dst_root,
        fps=src.meta.fps,
        robot_type=src.meta.robot_type,
        features=raw_features,
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=os.cpu_count(),
        encoder_queue_maxsize=1000,
    )

    task_map = {int(row["task_index"]): task for task, row in src.meta.tasks.iterrows()}
    eps_df = src.meta.episodes.to_pandas()

    # Load all parquet files into one dataframe
    data_files = eps_df.groupby(["data/chunk_index", "data/file_index"]).size().index.tolist()
    parquet_frames = []
    for chunk_idx, file_idx in data_files:
        ep_start = eps_df[
            (eps_df["data/chunk_index"] == chunk_idx) & (eps_df["data/file_index"] == file_idx)
        ]["episode_index"].iloc[0]
        path = src.root / src.meta.get_data_file_path(ep_index=int(ep_start))
        parquet_frames.append(pd.read_parquet(path))
    df = pd.concat(parquet_frames).sort_values("index").reset_index(drop=True)

    state_arr = np.stack(df["observation.state"].tolist()).astype(np.float32)
    effector_arr = np.stack(df["observation.effector_target_translation"].tolist()).astype(np.float32)
    action_arr = np.stack(df["action"].tolist()).astype(np.float32)
    reward_arr = df["next.reward"].to_numpy(dtype=np.float32)
    task_idx_arr = df["task_index"].to_numpy()

    zero_action = np.zeros(2, dtype=np.float32)
    zero_reward = np.zeros(1, dtype=np.float32)

    vid_chunk_col = "videos/observation.images.rgb/chunk_index"
    vid_file_col = "videos/observation.images.rgb/file_index"
    ts_col = "videos/observation.images.rgb/from_timestamp"

    current_container = None
    current_vid_key = None

    for _, ep_row in tqdm(eps_df.iterrows(), total=len(eps_df), desc="episodes"):
        ep_idx = int(ep_row["episode_index"])
        ep_len = int(ep_row["length"])
        from_ts = float(ep_row[ts_col])
        vid_key = (int(ep_row[vid_chunk_col]), int(ep_row[vid_file_col]))

        if vid_key != current_vid_key:
            if current_container is not None:
                current_container.close()
            video_path = src.root / src.meta.get_video_file_path(
                ep_index=ep_idx, vid_key="observation.images.rgb"
            )
            current_container = av.open(str(video_path))
            current_vid_key = vid_key

        frames = read_episode_frames(current_container, from_ts, ep_len)

        ep_df_rows = df[df["episode_index"] == ep_idx]
        local_indices = ep_df_rows.index.to_numpy()
        task = task_map[int(task_idx_arr[local_indices[0]])]

        for j, li in enumerate(local_indices):
            dst.add_frame({
                "observation.images.rgb": frames[j],
                "observation.state": state_arr[li],
                "observation.effector_target_translation": effector_arr[li],
                "action": action_arr[li],
                "next.reward": reward_arr[li: li + 1],
                "next.done": np.zeros(1, dtype=bool),
                "task": task,
            })

        last_img = frames[-1]
        last_state = state_arr[local_indices[-1]]
        last_effector = effector_arr[local_indices[-1]]

        for i in range(args.k):
            dst.add_frame({
                "observation.images.rgb": last_img,
                "observation.state": last_state,
                "observation.effector_target_translation": last_effector,
                "action": zero_action,
                "next.reward": zero_reward,
                "next.done": np.array([i == args.k - 1], dtype=bool),
                "task": task,
            })

        dst.save_episode()

    if current_container is not None:
        current_container.close()

    print(f"Done. {src.num_episodes} episodes, {src.num_frames + src.num_episodes * args.k} total frames.")


if __name__ == "__main__":
    main()
