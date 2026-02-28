#!/usr/bin/env python3
"""Example custom benchmark model — uses the official CMI baseline.

Copy this file and modify ``__init__`` / ``score_batch`` to plug in your
own model while keeping the :class:`BenchmarkModelABC` contract.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import numpy as np

_custom_models_dir = Path(__file__).resolve().parent
_benchmark_dir     = _custom_models_dir.parent          # CMI-RewardBench/
if str(_benchmark_dir) not in sys.path:
    sys.path.insert(0, str(_benchmark_dir))

from core.model_abc import BenchmarkBatchInput, BenchmarkModelABC


class CMIRewardModelBaseline(BenchmarkModelABC):
    """Official CMI Reward Model baseline.

    Parameters
    ----------
    checkpoint:
        Path to the ``.safetensors`` checkpoint file.
    config:
        Path to ``config.yaml``.  Auto-detected if *None*.
    device:
        Torch device string (default ``"cuda:0"``).
    mode:
        ``"final"`` (chunk-based, recommended) or ``"standard"`` (sliding window).
    """

    def __init__(
        self,
        checkpoint: str,
        config: str | None = None,
        device: str = "cuda:0",
        mode: str = "final",
        sr: int = 24000,
    ) -> None:
        from baselines.inference import RewardModelInference

        self._sr   = sr
        self._impl = RewardModelInference(
            checkpoint=checkpoint,
            config=config,
            device=device,
            sr=sr,
            mode=mode,
        )

    @property
    def sr(self) -> int:
        return self._sr

    def score_batch(
        self,
        inputs: List[BenchmarkBatchInput],
        batch_size: int,
        max_dur: float,
        **kwargs: Any,
    ) -> np.ndarray:
        payload = [x.to_dict() for x in inputs]
        return self._impl.score_batch(
            inputs=payload,
            batch_size=batch_size,
            max_dur=max_dur,
            **kwargs,
        )
