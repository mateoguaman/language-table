"""Dummy low-level policy that broadcasts pink-noise actions to all envs."""

from __future__ import annotations

from typing import List

import numpy as np

from language_table.environments.constants import X_MAX, X_MIN, Y_MAX, Y_MIN


class PinkNoisePolicy:
    """Generate one temporally-correlated random action stream for every env.

    ``predict_chunk`` follows the same contract as the VLA policies: it returns
    one action chunk per environment. Each active environment receives an
    identical copy of the same pink-noise chunk.
    """

    def __init__(
        self,
        seed: int = 0,
        chunk_size: int = 1,
        action_clip: float = 0.1,
        num_rows: int = 8,
        white_mix: float = 0.35,
        zero_bias: float = 0.5,
        bias_target: tuple[float, float] = (0.0,0.0),
    ) -> None:
        self.seed = int(seed)
        self.chunk_size = max(1, int(chunk_size))
        self.action_clip = float(action_clip)
        self.num_rows = max(1, int(num_rows))
        self.white_mix = float(np.clip(white_mix, 0.0, 1.0))
        self.zero_bias = float(np.clip(zero_bias, 0.0, 1.0))
        self.bias_target = np.asarray(bias_target, dtype=np.float32)
        self._rng = np.random.default_rng(self.seed)
        self._rows = np.zeros((self.num_rows, 2), dtype=np.float32)
        self._counter = 0

    def reset(self, num_envs: int = 1) -> None:
        del num_envs
        self._rng = np.random.default_rng(self.seed)
        self._rows = self._rng.uniform(
            -self.action_clip,
            self.action_clip,
            size=(self.num_rows, 2),
        ).astype(np.float32)
        self._counter = 0

    def _sample_action(self) -> np.ndarray:
        # Voss-McCartney pink noise: update lower rows more often than higher
        # rows, then average rows to produce a correlated 2D action.
        # self._counter += 1
        # trailing_zeros = (self._counter & -self._counter).bit_length() - 1
        # row_count = min(trailing_zeros + 1, self.num_rows)
        # self._rows[:row_count] = self._rng.uniform(
        #     -self.action_clip,
        #     self.action_clip,
        #     size=(row_count, 2),
        # ).astype(np.float32)
        # pink = self._rows.mean(axis=0)
        # white = self._rng.uniform(
        #     -self.action_clip, self.action_clip, size=(2,)
        # ).astype(np.float32)
        # action = (1.0 - self.white_mix) * pink + self.white_mix * white
        # action = (1.0 - self.zero_bias) * action + self.zero_bias * self.bias_target
        # return np.clip(action, -self.action_clip, self.action_clip).astype(np.float32)
        action = self._rng.uniform(
            -self.action_clip, self.action_clip, size=(2,)
        ).astype(np.float32)
        return np.clip(action, -self.action_clip, self.action_clip).astype(np.float32)

    def _sample_chunk(self) -> np.ndarray:
        return np.stack(
            [self._sample_action() for _ in range(self.chunk_size)],
            axis=0,
        )

    def predict_chunk(self, goals, obs_list, active_mask) -> List[np.ndarray]:
        del goals, obs_list
        active_mask = np.asarray(active_mask, dtype=bool)
        chunk = self._sample_chunk()
        return [
            chunk.copy() if active else np.zeros((1, 2), dtype=np.float32)
            for active in active_mask
        ]

    def predict(self, goals, obs_list, active_mask):
        chunks = self.predict_chunk(goals, obs_list, active_mask)
        return [chunk[0] for chunk in chunks]
