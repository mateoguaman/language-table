"""
Task provider hook for `reward_type=custom`.

Generates per-env language instructions (on reset) and per-turn rewards
(at the end of each outer LLM turn) from the final text + image
observation. Replace with API-backed subclasses when ready; the call
sites in the LAVA env manager stay the same.

Provider selection: the LAVA env manager receives a concrete provider
instance. When wiring from ``reward_kwargs`` JSON, a reserved
``"provider"`` key names the class in ``TASK_PROVIDERS``; remaining
keys are forwarded to that class's constructor. See
``build_task_provider`` below.
"""

import inspect
from typing import Any, Dict, Optional

import numpy as np


class TaskProvider:
    """Base hook for custom task generation.

    Subclasses override ``.instruction()`` / ``.reward()`` with real
    (possibly batched/async) API calls. Each subclass defines its own
    ``__init__`` to declare its required parameters; the base class has
    no constructor parameters.
    """

    def instruction(
        self,
        text_obs: str,
        image: Optional[np.ndarray],
        env_idx: int,
    ) -> str:
        raise NotImplementedError

    def reward(
        self,
        text_obs: str,
        image: Optional[np.ndarray],
        env_idx: int,
    ) -> float:
        raise NotImplementedError


class DummyTaskProvider(TaskProvider):
    """Placeholder implementation: fixed instruction, zero reward."""

    def __init__(self, instruction_str: str = "push any block anywhere"):
        self._instruction_str = instruction_str

    def instruction(self, text_obs, image, env_idx):
        return self._instruction_str

    def reward(self, text_obs, image, env_idx):
        return 0.0


from .tetris_task_provider import TetrisTaskProvider  # noqa: E402


TASK_PROVIDERS: Dict[str, type] = {
    "DummyTaskProvider": DummyTaskProvider,
    "TetrisTaskProvider": TetrisTaskProvider,
}


def build_task_provider(
    reward_kwargs: Dict[str, Any],
    group_n: int = 1,
) -> TaskProvider:
    """Instantiate a ``TaskProvider`` from parsed ``reward_kwargs``.

    Expects a ``"provider"`` key naming an entry in ``TASK_PROVIDERS``.
    Remaining keys are forwarded to the provider's constructor.

    ``group_n`` (the meta-RL group size from the env pool) is injected
    into the provider's kwargs when its constructor declares a
    ``group_n`` parameter and the caller did not already pass one via
    ``reward_kwargs``.

    Raises
    ------
    ValueError
        If ``"provider"`` is missing or not registered in ``TASK_PROVIDERS``.
    """
    kwargs = dict(reward_kwargs)
    name = kwargs.pop("provider", None)
    if name is None:
        raise ValueError(
            "reward_type=custom requires reward_kwargs to include a "
            f"'provider' key. Available providers: {sorted(TASK_PROVIDERS)}"
        )
    if name not in TASK_PROVIDERS:
        raise ValueError(
            f"Unknown task provider {name!r}. "
            f"Available providers: {sorted(TASK_PROVIDERS)}"
        )
    cls = TASK_PROVIDERS[name]
    if "group_n" in inspect.signature(cls.__init__).parameters:
        kwargs.setdefault("group_n", group_n)
    return cls(**kwargs)
