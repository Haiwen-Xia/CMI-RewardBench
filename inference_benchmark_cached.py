#!/usr/bin/env python3
"""Benchmark inference with embedding cache.

Core idea:
- Use the official CMI baseline inference backend (training-consistent)
- Cache encoded audio embeddings by (audio_path, max_dur) to disk
- Reuse cached embeddings for A/B/ref audio across all samples

Outputs (same as inference_benchmark):
- results/<ckpt_tag>/results.jsonl
- results/<ckpt_tag>/metrics_*.csv/json (if run_evaluate)
- results/<ckpt_tag>/metadata.json
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger("benchmark.inference.cached")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class InferenceCachedArgs:
    checkpoint: str
    dataset_jsonl: str
    dataset_root: str
    results_root: str = str(Path(__file__).parent / "results")
    cache_root: str = str(Path(__file__).parent / "cache_embeddings")
    split: str = "test"
    subset: str = "all"
    device: str = "cuda:0"
    batch_size: int = 4
    max_dur: float = 30.0
    write_cache: bool = False
    run_evaluate: bool = True
    eval_yaml: Optional[str] = None
    output_dir: Optional[str] = None


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_audio_path(audio_path: Optional[str], dataset_root: str) -> Optional[str]:
    if not audio_path:
        return None
    if os.path.isabs(audio_path):
        return audio_path
    return os.path.normpath(os.path.join(dataset_root, audio_path))


def _to_benchmark_dataset(source: str) -> str:
    source = source or ""
    if "PAM" in source:
        return "PAM"
    if source in {"MusicEval", "SongEval"}:
        return "MusicEval"
    if "Music Arena" in source:
        return "MusicArena"
    if source == "cmi-arena-annotation":
        return "CMI-Pref"
    return "Unknown"


def _prepare_item(row: Dict[str, Any], row_index: int, dataset_root: str) -> Dict[str, Any]:
    source = row.get("source", "")
    return {
        "_row_index": row_index,
        "source": source,
        "benchmark_dataset": _to_benchmark_dataset(source),
        "audio_a_resolved": _resolve_audio_path(row.get("audio-path"), dataset_root),
        "audio_b_resolved": _resolve_audio_path(row.get("audio2"), dataset_root),
        "ref_audio_resolved": _resolve_audio_path(row.get("ref-audio-path"), dataset_root),
        "prompt": row.get("prompt", "") or "",
        "lyrics": row.get("lyrics", "") or "",
        "raw": row,
    }


def _split_four_datasets(rows: List[Dict[str, Any]], dataset_root: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "PAM": [],
        "MusicEval": [],
        "MusicArena": [],
        "CMI-Pref": [],
    }
    for i, row in enumerate(rows):
        item = _prepare_item(row=row, row_index=i, dataset_root=dataset_root)
        ds = item["benchmark_dataset"]
        if ds in grouped:
            grouped[ds].append(item)
    return grouped


def _pair_pref_label(score_a: Optional[float], score_b: Optional[float]) -> Optional[str]:
    if score_a is None or score_b is None:
        return None
    if score_a > score_b:
        return "model_a"
    if score_b > score_a:
        return "model_b"
    return "both"


def _ckpt_tag(checkpoint: str) -> str:
    ckpt_path = Path(checkpoint).resolve()
    parts = ckpt_path.parts
    if len(parts) >= 3:
        return f"{parts[-3]}_{parts[-1]}_{datetime.now().strftime('%d_%H%M')}"
    return ckpt_path.name


def _resolve_run_dir(args: InferenceCachedArgs) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.results_root) / _ckpt_tag(args.checkpoint)


def _pad_seq_embeddings(embeds_list: List[torch.Tensor], masks_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(e.shape[0] for e in embeds_list)
    dim = embeds_list[0].shape[1]
    batch_size = len(embeds_list)

    padded_embeds = torch.zeros(batch_size, max_len, dim, dtype=embeds_list[0].dtype)
    padded_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (embeds, mask) in enumerate(zip(embeds_list, masks_list)):
        seq_len = embeds.shape[0]
        padded_embeds[i, :seq_len] = embeds
        padded_masks[i, :seq_len] = mask

    return padded_embeds, padded_masks


def _cache_key(audio_path: str, max_dur: float) -> str:
    src = f"{audio_path}|{max_dur}"
    return hashlib.md5(src.encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, cache_root: str, write_enabled: bool = False):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.mem: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.write_enabled = write_enabled

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.mem:
            self.hits += 1
            return self.mem[key]

        p = self.cache_root / f"{key}.pt"
        if p.exists():
            try:
                obj = torch.load(p, map_location="cpu", weights_only=False)
                self.mem[key] = obj
                self.hits += 1
                return obj
            except Exception as e:
                logger.warning(f"Corrupted cache file detected, removing and recomputing: {p} ({e})")
                try:
                    p.unlink(missing_ok=True)
                except Exception as rm_e:
                    logger.warning(f"Failed to remove corrupted cache file {p}: {rm_e}")

        self.misses += 1
        return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if not self.write_enabled:
            return

        self.mem[key] = value
        p = self.cache_root / f"{key}.pt"
        tmp = self.cache_root / f"{key}.pt.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        torch.save(value, tmp)
        os.replace(tmp, p)


def _build_zero_ref(dim: int) -> Dict[str, Any]:
    return {
        "audio_embeds": torch.zeros(2, dim, dtype=torch.float32),
        "audio_mask": torch.zeros(2, dtype=torch.bool),
        "valid": False,
        "error": "",
    }


def _encode_single_audio_cached(
    inference_model,
    cache: EmbeddingCache,
    audio_path: Optional[str],
    max_dur: float,
) -> Dict[str, Any]:
    if not audio_path:
        return {
            "audio_embeds": None,
            "audio_mask": None,
            "valid": False,
            "error": "audio_not_provided",
        }

    if not os.path.exists(audio_path):
        return {
            "audio_embeds": None,
            "audio_mask": None,
            "valid": False,
            "error": f"audio_not_found:{audio_path}",
        }

    key = _cache_key(audio_path, max_dur)
    cached = cache.get(key)
    if cached is not None:
        return {
            "audio_embeds": cached["audio_embeds"],
            "audio_mask": cached["audio_mask"],
            "valid": True,
            "error": "",
        }

    waveform = inference_model._load_waveform(audio_path, "audio", max_dur=max_dur)
    embeds_list, masks_list = inference_model._encode_full_audio_by_chunks([waveform], batch_size=1)
    embeds_cpu = embeds_list[0].detach().cpu().float()
    mask_cpu = masks_list[0].detach().cpu().bool()

    cache.put(key, {"audio_embeds": embeds_cpu, "audio_mask": mask_cpu})
    return {
        "audio_embeds": embeds_cpu,
        "audio_mask": mask_cpu,
        "valid": True,
        "error": "",
    }


def _score_side_cached(
    inference_model,
    cache: EmbeddingCache,
    rows: List[Dict[str, Any]],
    side_key: str,
    batch_size: int,
    max_dur: float,
    desc_prefix: str,
) -> Dict[int, Dict[str, Any]]:
    outputs: Dict[int, Dict[str, Any]] = {}
    target_indices = [i for i, r in enumerate(rows) if r.get(side_key) is not None]

    for start in tqdm(range(0, len(target_indices), batch_size), desc=f"{desc_prefix}:{side_key}"):
        batch_idx = target_indices[start:start + batch_size]

        texts: List[str] = []
        lyrics: List[str] = []
        eval_embeds_list: List[torch.Tensor] = []
        eval_masks_list: List[torch.Tensor] = []
        ref_embeds_list: List[Optional[torch.Tensor]] = []
        ref_masks_list: List[Optional[torch.Tensor]] = []
        metas: List[Tuple[int, bool, str]] = []

        dim_hint: Optional[int] = None

        for idx in batch_idx:
            row = rows[idx]
            eval_obj = _encode_single_audio_cached(
                inference_model=inference_model,
                cache=cache,
                audio_path=row.get(side_key),
                max_dur=max_dur,
            )

            ref_obj = _encode_single_audio_cached(
                inference_model=inference_model,
                cache=cache,
                audio_path=row.get("ref_audio_resolved"),
                max_dur=max_dur,
            ) if row.get("ref_audio_resolved") else {
                "audio_embeds": None,
                "audio_mask": None,
                "valid": True,
                "error": "",
            }

            valid = bool(eval_obj["valid"])
            err = "|".join([x for x in [eval_obj.get("error", ""), ref_obj.get("error", "")] if x])

            if valid:
                eval_emb = eval_obj["audio_embeds"]
                eval_msk = eval_obj["audio_mask"]
                eval_embeds_list.append(eval_emb)
                eval_masks_list.append(eval_msk)
                dim_hint = eval_emb.shape[1]
            else:
                # temporary placeholder, will overwrite after dim is known
                eval_embeds_list.append(torch.zeros(2, 768))
                eval_masks_list.append(torch.zeros(2, dtype=torch.bool))

            ref_embeds_list.append(ref_obj["audio_embeds"])
            ref_masks_list.append(ref_obj["audio_mask"])
            texts.append(row.get("prompt", "") or "")
            lyrics.append(row.get("lyrics", "") or "")
            metas.append((idx, valid, err))

        if dim_hint is None:
            for idx, valid, err in metas:
                outputs[idx] = {
                    "alignment": None,
                    "musicality": None,
                    "valid": False,
                    "error": f"{err}|no_valid_audio_in_batch",
                }
            continue

        # Fix invalid placeholders to correct dimension
        for i, (_, valid, _) in enumerate(metas):
            if not valid:
                eval_embeds_list[i] = torch.zeros(2, dim_hint, dtype=torch.float32)
                eval_masks_list[i] = torch.zeros(2, dtype=torch.bool)

        eval_embeds, eval_masks = _pad_seq_embeddings(eval_embeds_list, eval_masks_list)
        eval_embeds = eval_embeds.to(inference_model.device)
        eval_masks = eval_masks.to(inference_model.device)

        has_ref = any(x is not None for x in ref_embeds_list)
        if has_ref:
            fixed_ref_embeds: List[torch.Tensor] = []
            fixed_ref_masks: List[torch.Tensor] = []
            zero_ref = _build_zero_ref(dim_hint)
            for emb, msk in zip(ref_embeds_list, ref_masks_list):
                if emb is None or msk is None:
                    fixed_ref_embeds.append(zero_ref["audio_embeds"])
                    fixed_ref_masks.append(zero_ref["audio_mask"])
                else:
                    fixed_ref_embeds.append(emb)
                    fixed_ref_masks.append(msk)
            ref_embeds, ref_masks = _pad_seq_embeddings(fixed_ref_embeds, fixed_ref_masks)
            ref_embeds = ref_embeds.to(inference_model.device)
            ref_masks = ref_masks.to(inference_model.device)
        else:
            ref_embeds, ref_masks = None, None

        try:
            use_amp = torch.cuda.is_available() and str(inference_model.device).startswith("cuda")
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = inference_model.model.forward_raw_text(
                        prompt_texts=texts,
                        prompt_lyrics=lyrics,
                        prompt_audio_embeds=ref_embeds,
                        prompt_audio_mask=ref_masks,
                        eval_audio_embeds=eval_embeds,
                        eval_audio_mask=eval_masks,
                        return_embeddings=False,
                    )
            else:
                result = inference_model.model.forward_raw_text(
                    prompt_texts=texts,
                    prompt_lyrics=lyrics,
                    prompt_audio_embeds=ref_embeds,
                    prompt_audio_mask=ref_masks,
                    eval_audio_embeds=eval_embeds,
                    eval_audio_mask=eval_masks,
                    return_embeddings=False,
                )

            scores = result["scores"].float() if isinstance(result, dict) else result.float()

            for j, (row_idx, valid, err) in enumerate(metas):
                if valid:
                    outputs[row_idx] = {
                        "alignment": float(scores[j, 0].item()),
                        "musicality": float(scores[j, 1].item()),
                        "valid": True,
                        "error": err,
                    }
                else:
                    outputs[row_idx] = {
                        "alignment": None,
                        "musicality": None,
                        "valid": False,
                        "error": err,
                    }
        except Exception as e:
            for row_idx, valid, err in metas:
                outputs[row_idx] = {
                    "alignment": None,
                    "musicality": None,
                    "valid": False,
                    "error": f"{err}|batch_error:{e}",
                }

    return outputs


def main_func(args: InferenceCachedArgs) -> Dict[str, Any]:
    from baselines.inference import RewardModelInference
    inference_model = RewardModelInference(args.checkpoint, device=args.device)

    all_rows = _load_jsonl(args.dataset_jsonl)
    rows = [r for r in all_rows if str(r.get("split", "")) == args.split]
    logger.info(f"Loaded {len(all_rows)} rows, kept split={args.split}: {len(rows)}")

    run_dir = _resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_root)
    cache = EmbeddingCache(str(cache_dir), write_enabled=args.write_cache)

    out_rows: List[Dict[str, Any]] = []
    grouped = _split_four_datasets(rows=rows, dataset_root=args.dataset_root)

    all_subsets = ["PAM", "MusicEval", "MusicArena", "CMI-Pref"]
    selected_subsets = all_subsets if args.subset == "all" else [args.subset]
    logger.info(f"Selected subset(s): {selected_subsets}")

    for ds_name in selected_subsets:
        subset = grouped[ds_name]
        if not subset:
            continue
        logger.info(f"Infer dataset={ds_name}, size={len(subset)}")

        pred_a = _score_side_cached(
            inference_model=inference_model,
            cache=cache,
            rows=subset,
            side_key="audio_a_resolved",
            batch_size=args.batch_size,
            max_dur=args.max_dur,
            desc_prefix=ds_name,
        )
        pred_b = _score_side_cached(
            inference_model=inference_model,
            cache=cache,
            rows=subset,
            side_key="audio_b_resolved",
            batch_size=args.batch_size,
            max_dur=args.max_dur,
            desc_prefix=ds_name,
        )

        for i, item in enumerate(subset):
            base = dict(item["raw"])
            pa = pred_a.get(i, {})
            pb = pred_b.get(i, {})

            align_a = pa.get("alignment")
            mus_a = pa.get("musicality")
            align_b = pb.get("alignment")
            mus_b = pb.get("musicality")

            base["benchmark_dataset"] = item["benchmark_dataset"]
            base["predicted_text-music alignment"] = align_a
            base["predicted_musicality"] = mus_a

            if align_b is not None:
                base["predicted_text-music alignment_b"] = align_b
            if mus_b is not None:
                base["predicted_musicality_b"] = mus_b

            pref_align = _pair_pref_label(align_a, align_b)
            pref_music = _pair_pref_label(mus_a, mus_b)
            overall_a = None if align_a is None or mus_a is None else (align_a + mus_a)
            overall_b = None if align_b is None or mus_b is None else (align_b + mus_b)
            pref_overall = _pair_pref_label(overall_a, overall_b)

            if pref_overall is not None:
                base["predicted_preference"] = pref_overall
            if pref_music is not None:
                base["predicted_preference-musicality"] = pref_music
            if pref_align is not None:
                base["predicted_preference-alignment"] = pref_align

            base["inference_valid_a"] = bool(pa.get("valid", False))
            if item["audio_b_resolved"] is not None:
                base["inference_valid_b"] = bool(pb.get("valid", False))
            err = "|".join([x for x in [pa.get("error", ""), pb.get("error", "")] if x])
            base["inference_error"] = err

            out_rows.append(base)

    results_jsonl = run_dir / "results.jsonl"
    _save_jsonl(str(results_jsonl), out_rows)
    logger.info(f"Saved inference results: {results_jsonl}")

    eval_report = None
    if args.run_evaluate:
        from evaluate_results import EvaluateArgs, evaluate_from_rows

        eval_args = EvaluateArgs(results_jsonl_abs="", output_dir_abs=str(run_dir), eval_yaml_abs=args.eval_yaml)
        eval_report = evaluate_from_rows(rows=out_rows, args=eval_args)

    metadata = {
        "inference_mode": "final_cached",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "write_cache": args.write_cache,
        "split": args.split,
        "subset": args.subset,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_dur": args.max_dur,
        "run_dir": str(run_dir.resolve()),
        "results_jsonl": str(results_jsonl.resolve()),
        "num_rows": len(out_rows),
        "eval_report": eval_report,
    }
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")

    return {
        "num_rows": len(out_rows),
        "run_dir": str(run_dir),
        "results_jsonl": str(results_jsonl),
        "metadata_json": str(metadata_path),
        "eval_report": eval_report,
    }


def _parse_args() -> InferenceCachedArgs:
    parser = argparse.ArgumentParser(description="Benchmark inference with embedding cache")
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--results_root", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--cache_root", default=str(Path(__file__).parent / "cache_embeddings"))
    parser.add_argument("--output_dir", default=None, help="Override run dir; default is results_root/ckpt_tag")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all", "PAM", "MusicEval", "MusicArena", "CMI-Pref"],
        help="Infer a single benchmark subset or all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_dur", type=float, default=120.0)
    parser.add_argument("--write_cache", action="store_true", default=False, help="Enable writing new cache files. Default: read-only cache.")
    parser.add_argument("--run_evaluate", dest="run_evaluate", action="store_true", default=True)
    parser.add_argument("--no_run_evaluate", dest="run_evaluate", action="store_false")
    parser.add_argument("--eval_yaml", default=None)
    ns = parser.parse_args()
    if not ns.eval_yaml:
        ns.eval_yaml = str(Path(__file__).parent / "config" / "eval_benchmark.yaml")
    return InferenceCachedArgs(**vars(ns))


def main() -> None:
    args = _parse_args()
    main_func(args)


if __name__ == "__main__":
    main()
