#!/usr/bin/env python3
"""CMI Reward Model — Official Baseline Inference

Unified inference wrapper that supports two audio-encoding modes:

  ``mode='final'``     Training-consistent chunk encoding (recommended).
                       Audio is split into non-overlapping 30 s chunks, each
                       encoded separately; all chunk embeddings are
                       concatenated before the joint transformer.

  ``mode='standard'``  Encode a single segment (with optional sliding window).
                       Suitable for ablations or short audio.

Checkpoint layout (expected directory)::

    checkpoint_dir/
    ├── model.safetensors
    └── config.yaml

Usage::

    from baselines.inference import RewardModelInference

    model = RewardModelInference("path/to/model.safetensors")
    scores = model.score("song.mp3", text="A cheerful pop song")
    # {'alignment': 0.72, 'quality': 0.85}

Path notes
----------
This module adds ``<CMI-RewardBench>/models/cmi-rm/src`` to ``sys.path``
so that MuQ model modules are importable.

If your layout differs, set ``CMI_RM_SRC`` to the directory that contains
the ``muq`` package.
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from safetensors.torch import load_file as load_safetensors_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_baselines_dir = Path(__file__).resolve().parent          # .../CMI-RewardBench/baselines/
_repo_root     = _baselines_dir.parent                    # .../CMI-RewardBench/
_default_src   = _repo_root / "models" / "cmi-rm" / "src"

_model_src_root = Path(
    os.environ.get("CMI_RM_SRC", str(_default_src))
)
if str(_model_src_root) not in sys.path:
    sys.path.insert(0, str(_model_src_root))

def _get_model_utils():
    mod = importlib.import_module("muq.muq_mulan.utils.model_utils")
    return mod.ModelConfig, mod.create_model_from_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = 24000, max_dur: Optional[float] = None) -> torch.Tensor:
    """Load audio file → mono 1-D waveform tensor, optionally cropped."""
    waveform, orig_sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if orig_sr != sr:
        waveform = torchaudio.transforms.Resample(orig_sr, sr)(waveform)
    if max_dur is not None:
        waveform = waveform[: int(max_dur * sr)]
    return waveform


def _ensure_1d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.squeeze(0) if x.shape[0] == 1 else x.mean(dim=0)
    raise ValueError(f"Expected 1-D or 2-D waveform, got shape {tuple(x.shape)}")


def _pad_waveforms(waveforms: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.zeros(len(waveforms), max_len)
    mask    = torch.zeros(len(waveforms), max_len, dtype=torch.bool)
    for i, w in enumerate(waveforms):
        padded[i, : w.shape[0]] = w
        mask[i,   : w.shape[0]] = True
    return padded, mask


def _pad_embed_seqs(
    embeds: List[torch.Tensor],
    masks:  List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(e.shape[0] for e in embeds)
    dim     = embeds[0].shape[-1]
    out_e   = torch.zeros(len(embeds), max_len, dim, dtype=embeds[0].dtype)
    out_m   = torch.zeros(len(embeds), max_len,      dtype=torch.bool)
    for i, (e, m) in enumerate(zip(embeds, masks)):
        out_e[i, : e.shape[0]] = e
        out_m[i, : m.shape[0]] = m
    return out_e, out_m


def _split_chunks(waveform: torch.Tensor, sr: int, chunk_sec: float = CHUNK_SECONDS) -> List[torch.Tensor]:
    chunk_samples = int(chunk_sec * sr)
    if chunk_samples <= 0 or waveform.numel() == 0:
        return [waveform]
    chunks = [
        waveform[start : start + chunk_samples]
        for start in range(0, waveform.shape[0], chunk_samples)
        if waveform[start : start + chunk_samples].numel() > 0
    ]
    return chunks or [waveform]


def _sliding_windows(
    waveform: torch.Tensor,
    sr: int,
    max_dur: float,
    dur_step: Optional[float],
) -> List[torch.Tensor]:
    max_samples  = int(max_dur * sr)
    total        = waveform.shape[0]
    if dur_step is None or total <= max_samples:
        return [waveform[:max_samples] if total > max_samples else waveform]
    step_samples = int(dur_step * sr)
    windows, start = [], 0
    while start < total:
        seg = waveform[start : start + max_samples]
        if seg.shape[0] >= sr:          # require at least 1 s
            windows.append(seg)
        if start + max_samples >= total:
            break
        start += step_samples
    return windows or [waveform]


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_state_dict(ckpt_path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load model state dict from ``.safetensors`` checkpoint."""
    if ckpt_path.suffix != ".safetensors":
        raise ValueError(f"Checkpoint must be .safetensors, got: {ckpt_path}")
    return load_safetensors_file(str(ckpt_path), device=device)


def _find_config(ckpt_path: Path) -> Path:
    """Search for config.yaml adjacent to or one level above the checkpoint."""
    for candidate in (
        ckpt_path.parent / "config.yaml",
        ckpt_path.parent.parent / "config.yaml",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"config.yaml not found near {ckpt_path}. "
        "Pass config= explicitly or place config.yaml next to the checkpoint."
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RewardModelInference:
    """CMI Reward Model inference (official baseline).

    Parameters
    ----------
    checkpoint:
        Path to the ``.safetensors`` checkpoint file.
    config:
        Path to ``config.yaml``.  Auto-detected if *None*.
    device:
        Torch device string (``"cuda:0"``, ``"cpu"``, …).
    sr:
        Audio sample rate; must match the training config (default 24 000).
    mode:
        ``"final"``    — chunk-encode then concat (training-consistent).
        ``"standard"`` — encode single segment or sliding window.
    bf16:
        Enable bfloat16 autocast during inference (ignored on CPU).
    """

    def __init__(
        self,
        checkpoint: str,
        config: Optional[str] = None,
        device: str = "cuda:0",
        sr: int = 24000,
        mode: str = "final",
        bf16: bool = True,
    ) -> None:
        if mode not in ("final", "standard"):
            raise ValueError(f"mode must be 'final' or 'standard', got {mode!r}")

        self.device = device
        self.sr     = sr
        self.mode   = mode
        self.bf16   = bf16 and ("cuda" in device)

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if ckpt_path.suffix != ".safetensors":
            raise ValueError("checkpoint must be a .safetensors file")

        cfg_path = Path(config) if config else _find_config(ckpt_path)
        logger.info("Config:     %s", cfg_path)
        logger.info("Checkpoint: %s", ckpt_path)
        logger.info("Mode:       %s  |  bf16=%s", mode, self.bf16)

        ModelConfig, create_model_from_config = _get_model_utils()
        model_config = ModelConfig.from_yaml(str(cfg_path))
        model_config.null_embedding_config.skip_null = True

        self.model = create_model_from_config(model_config)

        state_dict             = _load_state_dict(ckpt_path)
        missing, unexpected    = self.model.load_state_dict(state_dict, strict=False)
        n_loaded               = len(state_dict) - len(unexpected)
        logger.info(
            "Loaded %d/%d weights  (missing=%d, unexpected=%d)",
            n_loaded, len(state_dict), len(missing), len(unexpected),
        )

        if mode == "final" and not hasattr(self.model, "audio_module"):
            raise AttributeError(
                "Model has no audio_module; cannot use mode='final'."
            )

        self.model.to(device).eval()
        logger.info("Model ready on %s", device)

    def _load_waveform(
        self,
        x: Union[str, torch.Tensor],
        field_name: str = "audio",
        max_dur: Optional[float] = 30.0,
    ) -> torch.Tensor:
        if isinstance(x, str):
            return load_audio(x, sr=self.sr, max_dur=max_dur)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{field_name} must be path or torch.Tensor, got {type(x)}")
        wav = _ensure_1d(x)
        if max_dur is not None:
            wav = wav[: int(max_dur * self.sr)]
        return wav

    def _encode_full_audio_by_chunks(
        self,
        waveforms: List[torch.Tensor],
        batch_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return self._chunk_encode(waveforms, encode_batch=batch_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        audio: Union[str, torch.Tensor],
        text: str = "",
        lyrics: str = "",
        ref_audio: Optional[Union[str, torch.Tensor]] = None,
        max_dur: float = 30.0,
        dur_step: Optional[float] = None,
    ) -> Dict[str, float]:
        """Score a single audio file.

        Returns
        -------
        dict with keys ``'alignment'`` and ``'quality'``.
        """
        arr = self.score_batch(
            [{"audio": audio, "text": text, "lyrics": lyrics, "ref_audio": ref_audio}],
            batch_size=1, max_dur=max_dur, dur_step=dur_step, show_progress=False,
        )
        return {"alignment": float(arr[0, 0]), "quality": float(arr[0, 1])}

    def score_batch(
        self,
        inputs: List[Dict[str, Any]],
        batch_size: int = 4,
        max_dur: float = 30.0,
        dur_step: Optional[float] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Score a list of items.

        Each item is a dict with keys:

        * ``audio``     (required) — file path or waveform tensor
        * ``text``      (optional) — text prompt
        * ``lyrics``    (optional) — lyrics string
        * ``ref_audio`` (optional) — reference audio path or tensor

        Returns
        -------
        ``np.ndarray`` of shape ``[N, 2]``: column 0 = alignment, column 1 = quality.
        """
        if not inputs:
            return np.zeros((0, 2), dtype=np.float32)
        if self.mode == "final":
            return self._score_final(inputs, batch_size, max_dur, show_progress)
        return self._score_standard(inputs, batch_size, max_dur, dur_step, show_progress)

    # Compatibility alias used by benchmark pipeline
    inference_batch = score_batch

    # ------------------------------------------------------------------
    # mode='final'
    # ------------------------------------------------------------------

    def _score_final(
        self,
        inputs: List[Dict],
        batch_size: int,
        max_dur: float,
        show_progress: bool,
    ) -> np.ndarray:
        all_scores: List[torch.Tensor] = []
        it = range(0, len(inputs), batch_size)
        if show_progress:
            it = tqdm(it, desc="Scoring (final)")
        for start in it:
            group = inputs[start : start + batch_size]
            all_scores.append(self._forward_final_group(group, max_dur).cpu())
        return torch.cat(all_scores, dim=0).numpy()

    @torch.no_grad()
    def _forward_final_group(self, group: List[Dict], max_dur: float) -> torch.Tensor:
        texts  = [g.get("text",   "") for g in group]
        lyrics = [g.get("lyrics", "") for g in group]

        # Eval audio: crop → chunk-encode → concat per sample
        eval_waves = [self._load_wave(g["audio"], max_dur) for g in group]
        e_eval_list, m_eval_list = self._chunk_encode(eval_waves)
        e_eval, m_eval = _pad_embed_seqs(e_eval_list, m_eval_list)
        e_eval = e_eval.to(self.device)
        m_eval = m_eval.to(self.device)

        # Ref audio (zeros if absent)
        has_ref = any(g.get("ref_audio") is not None for g in group)
        if has_ref:
            ref_waves = [
                self._load_wave(g["ref_audio"], max_dur)
                if g.get("ref_audio") is not None
                else torch.zeros(self.sr)
                for g in group
            ]
            e_ref_list, m_ref_list = self._chunk_encode(ref_waves)
            e_ref, m_ref = _pad_embed_seqs(e_ref_list, m_ref_list)
            e_ref = e_ref.to(self.device)
            m_ref = m_ref.to(self.device)
        else:
            e_ref = torch.zeros(*e_eval.shape[:2], e_eval.shape[2], device=self.device, dtype=e_eval.dtype)[:, :1]
            m_ref = torch.zeros(len(group), 1, dtype=torch.bool, device=self.device)

        with self._autocast():
            out = self.model.forward_raw_text(
                prompt_texts=texts,
                prompt_lyrics=lyrics,
                prompt_audio_embeds=e_ref,
                prompt_audio_mask=m_ref,
                eval_audio_embeds=e_eval,
                eval_audio_mask=m_eval,
            )
        return out["scores"]  # [B, 2]

    def _chunk_encode(
        self,
        waveforms: List[torch.Tensor],
        encode_batch: int = 32,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode each waveform by splitting into CHUNK_SECONDS chunks.

        Returns per-sample lists of (frame_embeds, frame_mask).
        """
        all_chunks: List[torch.Tensor] = []
        chunk_owner: List[int]         = []
        for i, wav in enumerate(waveforms):
            for chunk in _split_chunks(wav, self.sr, CHUNK_SECONDS):
                all_chunks.append(chunk)
                chunk_owner.append(i)

        embed_per_chunk: List[torch.Tensor] = []
        mask_per_chunk:  List[torch.Tensor] = []
        for start in range(0, len(all_chunks), encode_batch):
            batch_waves = all_chunks[start : start + encode_batch]
            padded, att = _pad_waveforms(batch_waves)
            e, m = self.model.audio_module(
                padded.to(self.device),
                attention_mask=att.to(self.device),
            )
            for j in range(e.shape[0]):
                valid_mask = m[j].cpu()
                embed_per_chunk.append(e[j].cpu()[valid_mask])   # keep valid frames only
                mask_per_chunk.append(torch.ones(valid_mask.sum(), dtype=torch.bool))

        # Reassemble: concatenate chunks belonging to the same sample
        n = len(waveforms)
        sample_e: List[List[torch.Tensor]] = [[] for _ in range(n)]
        sample_m: List[List[torch.Tensor]] = [[] for _ in range(n)]
        for idx, owner in enumerate(chunk_owner):
            sample_e[owner].append(embed_per_chunk[idx])
            sample_m[owner].append(mask_per_chunk[idx])

        dim = embed_per_chunk[0].shape[-1] if embed_per_chunk else 768
        result_e = [torch.cat(es) if es else torch.zeros(1, dim)      for es in sample_e]
        result_m = [torch.cat(ms) if ms else torch.ones(1, dtype=torch.bool) for ms in sample_m]
        return result_e, result_m

    # ------------------------------------------------------------------
    # mode='standard'
    # ------------------------------------------------------------------

    def _score_standard(
        self,
        inputs: List[Dict],
        batch_size: int,
        max_dur: float,
        dur_step: Optional[float],
        show_progress: bool,
    ) -> np.ndarray:
        # Expand inputs → (segment_dict, input_idx)
        segments:     List[Dict] = []
        seg_to_input: List[int]  = []

        prep_it = enumerate(inputs)
        if show_progress:
            prep_it = tqdm(list(prep_it), desc="Loading audio")
        for idx, inp in prep_it:
            wav     = self._load_wave(inp["audio"], None)      # load full duration
            windows = _sliding_windows(wav, self.sr, max_dur, dur_step)
            ref     = inp.get("ref_audio")
            ref_wav = self._load_wave(ref, max_dur) if ref is not None else None
            for win in windows:
                segments.append({
                    "wav":    win,
                    "text":   inp.get("text",   ""),
                    "lyrics": inp.get("lyrics", ""),
                    "ref_wav": ref_wav,
                })
                seg_to_input.append(idx)

        if not segments:
            return np.zeros((len(inputs), 2), dtype=np.float32)

        all_raw: List[torch.Tensor] = []
        score_it = range(0, len(segments), batch_size)
        if show_progress:
            score_it = tqdm(score_it, desc="Scoring (standard)")
        for start in score_it:
            batch = segments[start : start + batch_size]
            all_raw.append(self._forward_standard_batch(batch).cpu())

        raw = torch.cat(all_raw, dim=0)   # [total_segs, 2]

        # Average over sliding-window segments
        final  = torch.zeros(len(inputs), 2)
        counts = torch.zeros(len(inputs))
        for seg_idx, inp_idx in enumerate(seg_to_input):
            final[inp_idx]  += raw[seg_idx]
            counts[inp_idx] += 1
        counts.clamp_(min=1)
        return (final / counts.unsqueeze(-1)).numpy()

    @torch.no_grad()
    def _forward_standard_batch(self, batch: List[Dict]) -> torch.Tensor:
        texts  = [b["text"]   for b in batch]
        lyrics = [b["lyrics"] for b in batch]

        eval_padded, eval_att = _pad_waveforms([b["wav"] for b in batch])
        e_eval, m_eval = self.model.audio_module(
            eval_padded.to(self.device),
            attention_mask=eval_att.to(self.device),
        )

        has_ref = any(b.get("ref_wav") is not None for b in batch)
        if has_ref:
            ref_waves = [
                b["ref_wav"] if b.get("ref_wav") is not None else torch.zeros(self.sr)
                for b in batch
            ]
            ref_padded, ref_att = _pad_waveforms(ref_waves)
            e_ref, m_ref = self.model.audio_module(
                ref_padded.to(self.device),
                attention_mask=ref_att.to(self.device),
            )
        else:
            e_ref = torch.zeros_like(e_eval[:, :1])
            m_ref = torch.zeros(len(batch), 1, dtype=torch.bool, device=self.device)

        with self._autocast():
            out = self.model.forward_raw_text(
                prompt_texts=texts,
                prompt_lyrics=lyrics,
                prompt_audio_embeds=e_ref,
                prompt_audio_mask=m_ref,
                eval_audio_embeds=e_eval,
                eval_audio_mask=m_eval,
            )
        return out["scores"]  # [B, 2]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_wave(
        self,
        x: Union[str, torch.Tensor, None],
        max_dur: Optional[float],
    ) -> torch.Tensor:
        if x is None:
            return torch.zeros(self.sr, dtype=torch.float32)
        if isinstance(x, str):
            return load_audio(x, sr=self.sr, max_dur=max_dur)
        wav = _ensure_1d(x)
        if max_dur is not None:
            wav = wav[: int(max_dur * self.sr)]
        return wav

    def _autocast(self):
        """Return appropriate autocast context (bfloat16 on CUDA, no-op otherwise)."""
        if self.bf16:
            device_type = self.device.split(":")[0]   # 'cuda', 'cpu', …
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(
        description="Score AI-generated music with the CMI Reward Model baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inference.py -c model.safetensors -a song.mp3 -t "A happy pop song"
    python inference.py -c model.safetensors -a song.mp3 -t "Love song" -l "I love you..."
    python inference.py -c model.safetensors -a gen.mp3 -t "Jazz piece" -r ref.mp3 --mode standard
        """,
    )
    p.add_argument("-c", "--checkpoint", required=True, help="Path to model.safetensors")
    p.add_argument("--config",   default=None, help="Path to config.yaml (auto-detected if omitted)")
    p.add_argument("-a", "--audio",  required=True, help="Audio file to score")
    p.add_argument("-t", "--text",   default="",    help="Text prompt")
    p.add_argument("-l", "--lyrics", default="",    help="Lyrics (optional)")
    p.add_argument("-r", "--ref_audio", default=None, help="Reference audio (optional)")
    p.add_argument("--device",   default="cuda:0")
    p.add_argument("--mode",     default="final", choices=["final", "standard"])
    p.add_argument("--max_dur",  type=float, default=30.0, help="Max audio duration (s)")
    p.add_argument("--no_bf16",  action="store_true", help="Disable bfloat16 autocast")
    args = p.parse_args()

    model  = RewardModelInference(
        args.checkpoint, config=args.config,
        device=args.device, mode=args.mode, bf16=not args.no_bf16,
    )
    scores = model.score(
        audio=args.audio, text=args.text, lyrics=args.lyrics,
        ref_audio=args.ref_audio, max_dur=args.max_dur,
    )
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"Audio: {args.audio}")
    if args.text:
        print(f"Text:  {args.text}")
    print(sep)
    print(f"  Alignment Score: {scores['alignment']:.4f}")
    print(f"  Quality Score:   {scores['quality']:.4f}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
