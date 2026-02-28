#!/usr/bin/env python3
"""Benchmark inference on final_dataset_split.jsonl.

Design goals:
- Minimal dependencies (reuse RewardModelInference)
- Strictly use final_dataset_split format fields
- Default assumes audio paths in dataset are relative to dataset_root_abs
- Save unified results JSONL and optionally call evaluation by function
"""

import argparse
from datetime import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger("benchmark.inference")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from core.model_abc import BenchmarkBatchInput, BenchmarkModelABC


@dataclass
class InferenceArgs:
    checkpoint: str
    dataset_jsonl: str
    dataset_root: str
    results_root: str = str(Path(__file__).parent / "results")
    split: str = "test"
    subset: str = "all"
    reward_model_backend: str = "final"
    device: str = "cuda:0"
    batch_size: int = 4
    max_dur: float = 30.0
    model_init_kwargs: Optional[Dict[str, Any]] = None
    model_score_kwargs: Optional[Dict[str, Any]] = None
    run_evaluate: bool = True
    eval_yaml: Optional[str] = None
    output_dir: Optional[str] = None
    model_class_path: Optional[str] = None


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


def _parse_json_dict_arg(raw: Optional[str], arg_name: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{arg_name} must be a valid JSON object string, got: {raw}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"{arg_name} must decode to dict, got: {type(obj)}")
    return obj


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


def _build_inference_input(
    audio_path: Optional[str],
    ref_audio_path: Optional[str],
    text: str,
    lyrics: str,
    sr: int,
) -> Tuple[Any, bool, str]:
    ref_audio = None
    err_msgs: List[str] = []

    if ref_audio_path:
        if os.path.exists(ref_audio_path):
            ref_audio = ref_audio_path
        else:
            err_msgs.append(f"ref_audio_not_found:{ref_audio_path}")

    if audio_path and os.path.exists(audio_path):
        return BenchmarkBatchInput(
            audio=audio_path,
            text=text,
            lyrics=lyrics,
            ref_audio=ref_audio,
        ), True, "|".join(err_msgs)

    err_msgs.insert(0, f"audio_not_found:{audio_path}")
    return BenchmarkBatchInput(
        audio=torch.zeros(sr),
        text=text,
        lyrics=lyrics,
        ref_audio=ref_audio,
    ), False, "|".join(err_msgs)


def _predict_side(
    model: BenchmarkModelABC,
    rows: List[Dict[str, Any]],
    side_key: str,
    batch_size: int,
    max_dur: float,
    model_score_kwargs: Optional[Dict[str, Any]],
    desc_prefix: str,
) -> Dict[int, Dict[str, Any]]:
    """Predict one side (A or B) for all rows.

    Returns dict: row_index -> {alignment, musicality, valid, error}
    """
    outputs: Dict[int, Dict[str, Any]] = {}
    target_indices = [i for i, r in enumerate(rows) if r.get(side_key) is not None]

    for start in tqdm(range(0, len(target_indices), batch_size), desc=f"{desc_prefix}:{side_key}"):
        batch_idx = target_indices[start:start + batch_size]
        batch_inputs: List[Any] = []
        batch_meta: List[Tuple[int, bool, str]] = []

        for idx in batch_idx:
            row = rows[idx]
            inp, valid, err = _build_inference_input(
                audio_path=row.get(side_key),
                ref_audio_path=row.get("ref_audio_resolved"),
                text=row.get("prompt", "") or "",
                lyrics=row.get("lyrics", "") or "",
                sr=model.sr,
            )
            batch_inputs.append(inp)
            batch_meta.append((idx, valid, err))

        try:
            scores = model.score_batch(
                inputs=batch_inputs,
                batch_size=len(batch_inputs),
                max_dur=max_dur,
                **(model_score_kwargs or {}),
            )
            for j, (row_idx, valid, err) in enumerate(batch_meta):
                outputs[row_idx] = {
                    "alignment": float(scores[j, 0]),
                    "musicality": float(scores[j, 1]),
                    "valid": bool(valid),
                    "error": err,
                }
        except Exception as e:
            for row_idx, valid, err in batch_meta:
                outputs[row_idx] = {
                    "alignment": None,
                    "musicality": None,
                    "valid": False,
                    "error": f"{err}|batch_error:{e}",
                }

    return outputs


def _load_custom_model_class(class_path: str) -> Type[BenchmarkModelABC]:
    module_path, class_name = class_path.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")

    def _looks_like_benchmark_model(candidate_cls: Type[Any]) -> bool:
        has_sr = hasattr(candidate_cls, "sr")
        has_score_batch = callable(getattr(candidate_cls, "score_batch", None))
        return has_sr and has_score_batch

    is_valid = False
    try:
        is_valid = issubclass(cls, BenchmarkModelABC)
    except TypeError:
        is_valid = False

    if not is_valid:
        is_valid = _looks_like_benchmark_model(cls)

    if not is_valid:
        raise TypeError(
            f"{class_path} is not a valid benchmark model class. "
            "Expected BenchmarkModelABC subclass or class implementing `sr` and `score_batch`."
        )
    return cls


def _build_model(args: InferenceArgs) -> BenchmarkModelABC:
    if args.model_class_path:
        cls = _load_custom_model_class(args.model_class_path)
        init_kwargs = dict(args.model_init_kwargs or {})
        init_kwargs.setdefault("checkpoint", args.checkpoint)
        init_kwargs.setdefault("device", args.device)
        return cls(**init_kwargs)

    from core.model_interface import RewardModelAdapter, RewardModelAdapterConfig

    return RewardModelAdapter(
        RewardModelAdapterConfig(
            checkpoint=args.checkpoint,
            device=args.device,
            mode=args.reward_model_backend,
            init_kwargs=args.model_init_kwargs,
        )
    )


def _pair_pref_label(score_a: Optional[float], score_b: Optional[float]) -> Optional[str]:
    if score_a is None or score_b is None:
        return None
    if score_a > score_b:
        return "model_a"
    if score_b >= score_a:
        return "model_b"


def _ckpt_tag(checkpoint: str) -> str:
    ckpt_path = Path(checkpoint).resolve()
    parts = ckpt_path.parts
    if len(parts) >= 2:
        return f"{parts[-3]}_{parts[-1]}_{datetime.now().strftime('%d_%H%M')}"
    return ckpt_path.name


def _resolve_run_dir(args: InferenceArgs) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return Path(args.results_root) / _ckpt_tag(args.checkpoint)


def main_func(args: InferenceArgs) -> Dict[str, Any]:
    model = _build_model(args)

    all_rows = _load_jsonl(args.dataset_jsonl)
    rows = [r for r in all_rows if str(r.get("split", "")) == args.split]
    logger.info(f"Loaded {len(all_rows)} rows, kept split={args.split}: {len(rows)}")

    run_dir = _resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, Any]] = []
    grouped = _split_four_datasets(rows=rows, dataset_root=args.dataset_root)

    all_subsets = ["PAM", "MusicEval", "MusicArena", "CMI-Pref"]
    selected_subsets = all_subsets if args.subset == "all" else [args.subset]
    logger.info(f"Selected subset(s): {selected_subsets}")

    for ds_name in selected_subsets:
        subset = grouped[ds_name]# limit to 20 for quick testing; remove or increase for full inference
        if not subset:
            continue
        logger.info(f"Infer dataset={ds_name}, size={len(subset)}")

        pred_a = _predict_side(
            model=model,
            rows=subset,
            side_key="audio_a_resolved",
            batch_size=args.batch_size,
            max_dur=args.max_dur,
            model_score_kwargs=args.model_score_kwargs,
            desc_prefix=ds_name,
        )
        pred_b = _predict_side(
            model=model,
            rows=subset,
            side_key="audio_b_resolved",
            batch_size=args.batch_size,
            max_dur=args.max_dur,
            model_score_kwargs=args.model_score_kwargs,
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
        # try:
        #     from RewardModel.benchmark.evaluate_results import EvaluateArgs, evaluate_from_rows
        # except ModuleNotFoundError:
        from evaluate_results import EvaluateArgs, evaluate_from_rows

        eval_args = EvaluateArgs(results_jsonl_abs="", output_dir_abs=str(run_dir), eval_yaml_abs=args.eval_yaml)
        eval_report = evaluate_from_rows(rows=out_rows, args=eval_args)

    metadata = {
        "reward_model_backend": args.reward_model_backend,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_jsonl": str(Path(args.dataset_jsonl).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": args.split,
        "subset": args.subset,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_dur": args.max_dur,
        "model_init_kwargs": args.model_init_kwargs or {},
        "model_score_kwargs": args.model_score_kwargs or {},
        "model_class_path": args.model_class_path,
        "custom_model_init_checkpoint": args.checkpoint,
        "custom_model_init_device": args.device,
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


def _parse_args() -> InferenceArgs:
    parser = argparse.ArgumentParser(description="Benchmark inference prototype")
    parser.add_argument("--checkpoint", '-c', required=True)
    parser.add_argument("--dataset_jsonl", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--results_root", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--output_dir", default=None, help="Override run dir; default is results_root/ckpt_tag")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all", "PAM", "MusicEval", "MusicArena", "CMI-Pref"],
        help="Infer a single benchmark subset or all",
    )
    parser.add_argument(
        "--reward_model_backend",
        "--inference_mode",
        dest="reward_model_backend",
        default="final",
        choices=["standard", "final"],
        help="Default adapter mode: standard=sliding-window inference, final=chunk-based inference",
    )
    parser.add_argument(
        "--model_class_path",
        default=None,
        help="Optional custom model class path: module.submodule:ClassName (must inherit BenchmarkModelABC)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_dur", type=float, default=120.0)
    parser.add_argument(
        "--model_init_kwargs_json",
        default=None,
        help='Optional extra kwargs for custom model init only. checkpoint/device are passed directly from CLI. Example: {"sr":24000}',
    )
    parser.add_argument(
        "--model_score_kwargs_json",
        default=None,
        help='Extra kwargs for model.score_batch, JSON dict string. Example: {"some_flag": true}',
    )
    parser.add_argument("--run_evaluate", dest="run_evaluate", action="store_true", default=True)
    parser.add_argument("--no_run_evaluate", dest="run_evaluate", action="store_false")
    parser.add_argument("--eval_yaml", default=None)
    ns = parser.parse_args()
    if not ns.eval_yaml:
        ns.eval_yaml = str(Path(__file__).parent / "config" / "eval_benchmark.yaml")
    ns.model_init_kwargs = _parse_json_dict_arg(ns.model_init_kwargs_json, "model_init_kwargs_json")
    ns.model_score_kwargs = _parse_json_dict_arg(ns.model_score_kwargs_json, "model_score_kwargs_json")
    ns_vars = vars(ns)
    ns_vars.pop("model_init_kwargs_json", None)
    ns_vars.pop("model_score_kwargs_json", None)
    return InferenceArgs(**vars(ns))


def main() -> None:
    args = _parse_args()
    main_func(args)


if __name__ == "__main__":
    main()