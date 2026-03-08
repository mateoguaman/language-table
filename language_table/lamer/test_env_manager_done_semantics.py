import unittest

import numpy as np

from language_table.lamer.env_manager import LanguageTableEnvironmentManager


def _obs(task: str, x: float):
    encoded = np.array(list(task.encode("utf-8")) + [0] * 32, dtype=np.int32)
    return {
        "instruction": encoded,
        "effector_translation": np.array([x, 0.0], dtype=np.float32),
        "block_red_star_translation": np.array([0.1, 0.2], dtype=np.float32),
        "block_red_star_mask": np.array([1.0], dtype=np.float32),
    }


class _DummyPolicy:
    def reset(self, num_envs: int):
        self.num_envs = num_envs

    def predict(self, goals, obs_list, active_mask):
        return [
            np.array([0.01, -0.02], dtype=np.float32) if active_mask[i]
            else np.zeros(2, dtype=np.float32)
            for i in range(len(goals))
        ]


class _FakeVectorEnv:
    def __init__(self):
        self.num_processes = 2
        self.step_calls = [0, 0]

    def reset(self):
        return [_obs("task 0", 0.0), _obs("task 1", 1.0)], [{}, {}]

    def restart(self):
        return self.reset()

    def step(self, actions, active_mask=None, cached_obs=None, cached_infos=None):
        self.last_active_mask = np.array(active_mask, copy=True)

        obs_list = list(cached_obs)
        rewards = [0.0, 0.0]
        dones = [True, True]
        infos = [dict(info) for info in cached_infos]

        for idx in range(self.num_processes):
            if not active_mask[idx]:
                infos[idx]["skipped_step_after_done"] = True
                continue

            self.step_calls[idx] += 1
            obs_list[idx] = _obs(f"task {idx}", float(self.step_calls[idx]))
            rewards[idx] = float(self.step_calls[idx])
            dones[idx] = self.step_calls[idx] >= (1 if idx == 0 else 2)
            infos[idx]["step_count"] = self.step_calls[idx]

        return obs_list, rewards, dones, infos


class TestEnvManagerDoneSemantics(unittest.TestCase):
    def test_manager_does_not_step_done_envs_again(self):
        envs = _FakeVectorEnv()
        manager = LanguageTableEnvironmentManager(
            envs=envs,
            num_attempts=1,
            max_turns=1,
            max_inner_steps=3,
            vla_policy=_DummyPolicy(),
        )

        manager.reset()
        _, rewards, dones, infos = manager.step(
            ["goal 0", "goal 1"],
            phase="play",
        )

        self.assertEqual(envs.step_calls, [1, 2])
        self.assertEqual(dones.tolist(), [True, True])
        self.assertEqual(rewards.tolist(), [1.0, 3.0])
        self.assertEqual(infos[0]["step_count"], 1)
        self.assertEqual(infos[1]["step_count"], 2)


if __name__ == "__main__":
    unittest.main()
