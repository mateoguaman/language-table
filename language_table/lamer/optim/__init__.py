"""Optimization variants for LAVAPolicy._build_batch.

Each variant (v0–v6) is a self-contained BatchBuilder that implements the same
interface as the original _build_batch but with a different optimization
strategy.  This lets us benchmark and test each change independently before
combining the best ones.
"""

from .base import BatchBuilder, get_real_tokenizer  # noqa: F401
