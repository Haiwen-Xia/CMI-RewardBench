#!/usr/bin/env python3
"""Default model adapter for benchmark inference.

Abstract interface and canonical batch input are defined in ``model_abc.py``.
This module provides the default adapter that wraps the official CMI baseline
(``baselines/inference.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from core.model_abc import BenchmarkBatchInput, BenchmarkModelABC


BenchmarkModelInterface = BenchmarkModelABC


@dataclass
class RewardModelAdapterConfig:
    checkpoint: str
    config: Optional[str] = None          # path to config.yaml; auto-detected if None
    device: str = "cuda:0"
    mode: str = "final"                   # "final" or "standard"
    init_kwargs: Optional[Dict[str, Any]] = None


class RewardModelAdapter(BenchmarkModelABC):
    """Default adapter wrapping the official CMI baseline.

    ``mode="final"``    → chunk-encode (training-consistent, recommended)
    ``mode="standard"`` → encode full segment / sliding window
    """

    def __init__(self, cfg: RewardModelAdapterConfig) -> None:
        baseline_module = importlib.import_module("baselines.inference")
        RewardModelInference = baseline_module.RewardModelInference

        init_kwargs = dict(cfg.init_kwargs or {})
        self._impl = RewardModelInference(
            checkpoint=cfg.checkpoint,
            config=cfg.config,
            device=cfg.device,
            mode=cfg.mode,
            **init_kwargs,
        )

    @property
    def sr(self) -> int:
        return int(self._impl.sr)

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
