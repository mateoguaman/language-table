import cv2
import numpy as np
from language_table.environments import blocks, language_table
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards.block2absolutelocation import ABSOLUTE_LOCATIONS
from language_table.environments.oracles import push_oracle_rrt_slowdown
from tf_agents.environments import gym_wrapper
from tf_agents.trajectories import time_step as ts

env = language_table.LanguageTable(
    block_mode=blocks.LanguageTableBlockVariants.BLOCK_4,
    reward_factory=block2absolutelocation.BlockToAbsoluteLocationReward,
    control_frequency=10.0,
    seed=6
)

env = gym_wrapper.GymWrapper(env)

if not hasattr(env, "get_control_frequency"):
    env.get_control_frequency = lambda: env._control_frequency

oracle = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
    env, use_ee_planner=True)

raw_env = env.gym

# One location per block, sampled randomly (without replacement)
block_list = list(blocks.FIXED_4_COMBINATION)
location_list = list(ABSOLUTE_LOCATIONS.keys())
rng = np.random.default_rng(seed=5)
block_list = rng.choice(block_list, size=len(block_list), replace=False)
block_location_pairs = list(zip(block_list, rng.choice(location_list, size=len(block_list), replace=False)))

# Initial env reset — physics state is established once here
env.reset()


def switch_task(env, raw_env, block, location):
    """Reassign instruction and reward target without touching physics."""
    state = raw_env._compute_state(request_task_update=False)
    info = raw_env._reward_calculator.reset_to(
        state, block, location, raw_env._blocks_on_table
    )
    raw_env._set_task_info(info)
    instruction = language_table.LanguageTable.decode_instruction(raw_env._instruction)
    print(f"Task: {instruction}")
    # Build a FIRST-type timestep from the current (unchanged) observation and
    # inject it into the wrapper so env.step() doesn't trigger an auto-reset.
    obs = raw_env._compute_observation()
    restart_ts = ts.restart(obs)
    env._current_time_step = restart_ts
    env._done = False
    return restart_ts


for block, location in block_location_pairs:
    current_ts = switch_task(env, raw_env, block, location)
    oracle.reset()

    while not current_ts.is_last():
        step = oracle.action(current_ts, ())
        current_ts = env.step(step.action)

        frame = raw_env.render(mode="rgb_array")
        cv2.imshow("Language Table", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
