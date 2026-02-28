#!/usr/bin/env python3
"""Abstract benchmark model interface.

Third-party evaluators can implement this interface and plug into
`inference_benchmark.py` without touching benchmark pipeline logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import numpy as np


@dataclass
class BenchmarkBatchInput:
    """Canonical batch input format consumed by benchmark model interfaces."""

    audio: str | torch.Tensor
    text: str = ""
    lyrics: str = ""
    ref_audio: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio": self.audio,
            "text": self.text,
            "lyrics": self.lyrics,
            "ref_audio": self.ref_audio,
        }


class BenchmarkModelABC(ABC):
    """Minimal enforced contract for benchmark inference models.

    Required:
    - `sr` property
    - `score_batch` method
    """

    @property
    @abstractmethod
    def sr(self) -> int:
        pass

    @abstractmethod
    def score_batch(
        self,
        inputs: List[BenchmarkBatchInput],
        batch_size: int,
        max_dur: float,
        **kwargs: Any,
    ) -> np.ndarray:
        pass
