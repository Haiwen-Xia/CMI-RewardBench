#!/usr/bin/env python3
"""Benchmark evaluation for inference_benchmark outputs.

- Input: results.jsonl produced by inference_benchmark.py
- Output: pandas csv/json summaries

Required keys by dataset in results rows:
- PAM:
    - ground truth: `musicality`, `text-music alignment`
    - prediction: `predicted_musicality`, `predicted_text-music alignment`
- MusicEval:
    - ground truth: `musicality`
    - prediction: `predicted_musicality`
- MusicArena:
    - ground truth: `preference`
    - prediction: `predicted_preference`
- CMI-Pref:
    - ground truth: `preference-musicality`, `preference-alignment`
    - prediction: `predicted_preference-musicality`, `predicted_preference-alignment`
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

logger = logging.getLogger("benchmark.evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


_DURATION_MAP_PATH = Path('../../audio_duration_map.json')
_DURATION_MAP: Optional[Dict[str, float]] = None


def _get_duration_map() -> Dict[str, float]:
    global _DURATION_MAP
    if _DURATION_MAP is None:
        if _DURATION_MAP_PATH.exists():
            with open(_DURATION_MAP_PATH, "r", encoding="utf-8") as f:
                _DURATION_MAP = json.load(f)
            logger.info(f"Loaded duration map: {_DURATION_MAP_PATH} ({len(_DURATION_MAP)} entries)")
        else:
            logger.warning(f"Duration map not found: {_DURATION_MAP_PATH}")
            _DURATION_MAP = {}
    return _DURATION_MAP


@dataclass
class EvaluateArgs:
    results_jsonl_abs: str
    output_dir_abs: str
    eval_yaml_abs: Optional[str] = None


def _assert_abs(path_value: str, name: str) -> None:
    if not os.path.isabs(path_value):
        raise ValueError(f"{name} must be an absolute path: {path_value}")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_cfg(eval_yaml_abs: Optional[str]) -> Dict[str, Any]:
    if not eval_yaml_abs:
        raise ValueError("eval_yaml_abs is required for loading evaluation configuration.")
    
    with open(eval_yaml_abs, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}


    return user_cfg


def _norm_pref(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    mapping = {
        "a": "model_a",
        "b": "model_b",
        "model_a": "model_a",
        "model_b": "model_b",
        "both": "both",
        "neither": "both",
    }
    return mapping.get(s)


def _safe_corr(s1: pd.Series, s2: pd.Series, method: str) -> Optional[float]:
    if len(s1) < 2:
        return None
    try:
        if method == "spearman":
            return float(s1.corr(s2, method="spearman"))
        if method == "kendall":
            return float(s1.corr(s2, method="kendall"))
        return float(s1.corr(s2, method="pearson"))
    except Exception:
        return None


def _extract_month_from_audio_path(audio_path: Any) -> str:
    s = "" if audio_path is None else str(audio_path).strip()
    if not s:
        return "unknown"

    parts = Path(s).parts
    if "MusicArena_data" in parts:
        idx = parts.index("MusicArena_data")
        if idx + 2 < len(parts):
            return parts[idx + 2]

    return "unknown"


def _extract_month(row: pd.Series) -> str:
    month = _extract_month_from_audio_path(row.get("audio-path"))
    if month != "unknown":
        return month
    return _extract_month_from_audio_path(row.get("audio2"))


def _infer_benchmark_dataset(row: pd.Series) -> str:
    source = str(row.get("source", ""))

    if "PAM" in source:
        return "PAM"
    if source in {"MusicEval", "SongEval"}:
        return "MusicEval"
    if "Music Arena" in source:
        return "MusicArena"
    if source == "cmi-arena-annotation":
        return "CMI-Pref"

    # Fallback: infer MusicArena from audio path pattern
    if _extract_month(row) != "unknown":
        return "MusicArena"

    # Last resort: keep existing value only if already valid
    existing = str(row.get("benchmark_dataset", "")).strip()
    if existing in {"PAM", "MusicEval", "MusicArena", "CMI-Pref"}:
        return existing

    return "Unknown"


def _infer_modality(row: pd.Series) -> str:
    if "modality" in row and pd.notna(row["modality"]) and str(row["modality"]).strip():
        return str(row["modality"]).strip()
    has_lyrics = bool(str(row.get("lyrics", "")).strip())
    has_ref = bool(str(row.get("ref-audio-path", "")).strip())
    if has_lyrics and has_ref:
        return "text+lyrics+ref"
    if has_lyrics:
        return "text+lyrics"
    if has_ref:
        return "text+ref"
    return "text"


def _acc(df: pd.DataFrame, gt_col: str, pred_col: str) -> Dict[str, Any]:
    tmp = df[[gt_col, pred_col]].copy()
    tmp[gt_col] = tmp[gt_col].map(_norm_pref)
    tmp[pred_col] = tmp[pred_col].map(_norm_pref)
    valid_labels = ["model_a", "model_b", "both"]
    tmp = tmp[tmp[gt_col].isin(valid_labels) & tmp[pred_col].isin(valid_labels)]
    n = len(tmp)
    if n == 0:
        return {"n": 0, "acc": None}

    def _row_score(row: pd.Series) -> float:
        gt = row[gt_col]
        pred = row[pred_col]
        if gt == pred:
            return 1.0
        if gt == "both" or pred == "both":
            return 0.5
        return 0.0

    scores = tmp.apply(_row_score, axis=1)
    return {"n": int(n), "acc": float(scores.mean())}


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _fmt4(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.4f}"


def _fmt_pct(v: Optional[float]) -> str:
    """Format a [0,1] accuracy as XX.XX (percent, 2 decimal places)."""
    if v is None:
        return "-"
    return f"{v * 100:.2f}"


def _lookup_duration(audio_path: Any) -> Optional[float]:
    """Look up audio duration (seconds) from the pre-built JSON map."""
    if audio_path is None:
        return None
    path_str = str(audio_path).strip()
    if not path_str:
        return None

    dur_map = _get_duration_map()

    # Try exact key first (as stored in the JSONL, e.g. relative path)
    if path_str in dur_map:
        return float(dur_map[path_str])

    # Try forward-slash normalised key
    norm = path_str.replace("\\", "/")
    if norm in dur_map:
        return float(dur_map[norm])

    # Try filename-only as last resort
    fname = Path(path_str).name
    if fname in dur_map:
        return float(dur_map[fname])

    return None


def _row_duration_seconds(row: pd.Series) -> Optional[float]:
    audio_a = row.get("audio-path")
    if audio_a is None or not str(audio_a).strip():
        audio_a = row.get("audio")

    audio_b = row.get("audio2")

    dur_a = _lookup_duration(audio_a)
    dur_b = _lookup_duration(audio_b)

    if dur_a is None and dur_b is None:
        return None
    if dur_b is None:
        return dur_a
    if dur_a is None:
        return dur_b
    return float((dur_a + dur_b) / 2.0)


def _confidence_group(v: Any) -> Optional[str]:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return None
    if float(x) < 3:
        return "<3"
    if float(x) > 3:
        return ">3"
    return "=3"


def _duration_bin_label(v: Any, bins: List[float]) -> Optional[str]:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return None
    xv = float(x)
    if len(bins) < 2:
        return None

    for i in range(len(bins) - 1):
        left = float(bins[i])
        right = float(bins[i + 1])
        if i < len(bins) - 2:
            if left <= xv < right:
                return f"[{left:g},{right:g})"
        else:
            if left <= xv <= right:
                return f"[{left:g},{right:g}]"
    return None


def _compute_summary_rows(df: pd.DataFrame, include_lcc: bool, include_k_tau: bool) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []

    # PAM: Musicality + Alignment
    pam = df[df["benchmark_dataset"] == "PAM"] if "benchmark_dataset" in df.columns else df.iloc[0:0]
    if not pam.empty:
        for task_name, gt_col, pred_col in [
            ("musicality", "musicality", "predicted_musicality"),
            ("alignment", "text-music alignment", "predicted_text-music alignment"),
        ]:
            tmp = pam[[gt_col, pred_col]].copy()
            tmp[gt_col] = pd.to_numeric(tmp[gt_col], errors="coerce")
            tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
            tmp = tmp.dropna()
            srcc = _safe_corr(tmp[gt_col], tmp[pred_col], "spearman")
            row = {"dataset": "PAM", "task": task_name, "n": int(len(tmp)), "SRCC": srcc}
            if include_lcc:
                row["LCC"] = _safe_corr(tmp[gt_col], tmp[pred_col], "pearson")
            if include_k_tau:
                row["K-Tau"] = _safe_corr(tmp[gt_col], tmp[pred_col], "kendall")
            summary_rows.append(row)

    # MusicEval: Musicality
    me = df[df["benchmark_dataset"] == "MusicEval"] if "benchmark_dataset" in df.columns else df.iloc[0:0]
    if not me.empty:
        tmp = me[["musicality", "predicted_musicality"]].copy()
        tmp["musicality"] = pd.to_numeric(tmp["musicality"], errors="coerce")
        tmp["predicted_musicality"] = pd.to_numeric(tmp["predicted_musicality"], errors="coerce")
        tmp = tmp.dropna()
        row = {
            "dataset": "MusicEval",
            "task": "musicality",
            "n": int(len(tmp)),
            "SRCC": _safe_corr(tmp["musicality"], tmp["predicted_musicality"], "spearman"),
        }
        if include_lcc:
            row["LCC"] = _safe_corr(tmp["musicality"], tmp["predicted_musicality"], "pearson")
        if include_k_tau:
            row["K-Tau"] = _safe_corr(tmp["musicality"], tmp["predicted_musicality"], "kendall")
        summary_rows.append(row)

    # MusicArena: overall preference acc — use explicit predicted_preference field
    ma = df[df["benchmark_dataset"] == "MusicArena"] if "benchmark_dataset" in df.columns else df.iloc[0:0]
    if not ma.empty:
        acc_obj = _acc(ma, "preference", "predicted_preference")
        summary_rows.append({
            "dataset": "MusicArena",
            "task": "preference",
            "n": acc_obj["n"],
            "ACC": acc_obj["acc"],
        })

    # CMI-Pref: alignment/musicality acc
    cmi = df[df["benchmark_dataset"] == "CMI-Pref"] if "benchmark_dataset" in df.columns else df.iloc[0:0]
    if not cmi.empty:
        for task_name, gt_col, pred_col in [
            ("alignment", "preference-alignment", "predicted_preference-alignment"),
            ("musicality", "preference-musicality", "predicted_preference-musicality"),
        ]:
            acc_obj = _acc(cmi, gt_col, pred_col)
            summary_rows.append({
                "dataset": "CMI-Pref",
                "task": task_name,
                "n": acc_obj["n"],
                "ACC": acc_obj["acc"],
            })

    return summary_rows


def _build_temp_ordered_latex(summary_rows: List[Dict[str, Any]], detail_rows: List[Dict[str, Any]], cfg: Dict[str, Any], out_dir: Path) -> str:
    training_variant = str(cfg.get("training_variant", out_dir.name))

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    def _summary_val(dataset: str, task: str, col: str) -> Optional[float]:
        if summary_df.empty or col not in summary_df.columns:
            return None
        hit = summary_df[(summary_df["dataset"] == dataset) & (summary_df["task"] == task)]
        if hit.empty:
            return None
        v = hit.iloc[0].get(col)
        return None if pd.isna(v) else float(v)

    def _detail_acc(
        dataset: str,
        task: str,
        group_key: str,
        group_value: str,
        ref_category: Optional[str] = None,
    ) -> Optional[float]:
        if detail_df.empty or "ACC" not in detail_df.columns:
            return None
        mask = (
            (detail_df["dataset"] == dataset)
            & (detail_df["task"] == task)
            & (detail_df["group_key"] == group_key)
            & (detail_df["group_value"] == group_value)
        )
        if ref_category is not None and "ref_category" in detail_df.columns:
            mask = mask & (detail_df["ref_category"] == ref_category)
        hit = detail_df[mask]
        if hit.empty:
            return None
        v = hit.iloc[0].get("ACC")
        return None if pd.isna(v) else float(v)

    # ===== CMI-Pref breakdown lines =====
    # Musicality: Total | Conf<3 | =3 | >3 | Instru(no lyrics) | Vocal(lyrics)
    cmi_mus_total = _summary_val("CMI-Pref", "musicality", "ACC")
    cmi_mus_c_lt3 = _detail_acc("CMI-Pref", "musicality", "confidence", "<3", "all")
    cmi_mus_c_eq3 = _detail_acc("CMI-Pref", "musicality", "confidence", "=3", "all")
    cmi_mus_c_gt3 = _detail_acc("CMI-Pref", "musicality", "confidence", ">3", "all")
    cmi_mus_instru = _detail_acc("CMI-Pref", "musicality", "modality_type", "instru")
    cmi_mus_vocal  = _detail_acc("CMI-Pref", "musicality", "modality_type", "vocal")

    # Text-Music Alignment (no ref): Total | Conf<3 | =3 | >3 | Instru(text) | Vocal(text+lyrics)
    cmi_tma_total  = _detail_acc("CMI-Pref", "alignment", "ref_subset", "no_ref")
    cmi_tma_c_lt3  = _detail_acc("CMI-Pref", "alignment", "confidence", "<3",  "no_ref")
    cmi_tma_c_eq3  = _detail_acc("CMI-Pref", "alignment", "confidence", "=3",  "no_ref")
    cmi_tma_c_gt3  = _detail_acc("CMI-Pref", "alignment", "confidence", ">3",  "no_ref")
    cmi_tma_instru = _detail_acc("CMI-Pref", "alignment", "modality", "text")
    cmi_tma_vocal  = _detail_acc("CMI-Pref", "alignment", "modality", "text+lyrics")

    # Audio-Music Alignment (ref): Total | Conf<3 | =3 | >3 | Instru(text+ref) | Vocal(text+lyrics+ref)
    cmi_ama_total  = _detail_acc("CMI-Pref", "alignment", "ref_subset", "ref")
    cmi_ama_c_lt3  = _detail_acc("CMI-Pref", "alignment", "confidence", "<3",  "ref")
    cmi_ama_c_eq3  = _detail_acc("CMI-Pref", "alignment", "confidence", "=3",  "ref")
    cmi_ama_c_gt3  = _detail_acc("CMI-Pref", "alignment", "confidence", ">3",  "ref")
    cmi_ama_instru = _detail_acc("CMI-Pref", "alignment", "modality", "text+ref")
    cmi_ama_vocal  = _detail_acc("CMI-Pref", "alignment", "modality", "text+lyrics+ref")

    cmi_musicality_line = " & ".join([
        training_variant,
        _fmt_pct(cmi_mus_total),
        _fmt_pct(cmi_mus_c_lt3),
        _fmt_pct(cmi_mus_c_eq3),
        _fmt_pct(cmi_mus_c_gt3),
        _fmt_pct(cmi_mus_instru),
        _fmt_pct(cmi_mus_vocal),
    ]) + r" \\"

    cmi_tma_line = " & ".join([
        training_variant,
        _fmt_pct(cmi_tma_total),
        _fmt_pct(cmi_tma_c_lt3),
        _fmt_pct(cmi_tma_c_eq3),
        _fmt_pct(cmi_tma_c_gt3),
        _fmt_pct(cmi_tma_instru),
        _fmt_pct(cmi_tma_vocal),
    ]) + r" \\"

    cmi_ama_line = " & ".join([
        training_variant,
        _fmt_pct(cmi_ama_total),
        _fmt_pct(cmi_ama_c_lt3),
        _fmt_pct(cmi_ama_c_eq3),
        _fmt_pct(cmi_ama_c_gt3),
        _fmt_pct(cmi_ama_instru),
        _fmt_pct(cmi_ama_vocal),
    ]) + r" \\"

    # ===== Musicality block (ordered) =====
    pam_m_lcc = _summary_val("PAM", "musicality", "LCC")
    pam_m_srcc = _summary_val("PAM", "musicality", "SRCC")
    pam_m_ktau = _summary_val("PAM", "musicality", "K-Tau")

    me_lcc = _summary_val("MusicEval", "musicality", "LCC")
    me_srcc = _summary_val("MusicEval", "musicality", "SRCC")
    me_ktau = _summary_val("MusicEval", "musicality", "K-Tau")

    ma_acc = _summary_val("MusicArena", "preference", "ACC")
    cmi_music_acc = _summary_val("CMI-Pref", "musicality", "ACC")

    # ===== Alignment block (ordered) =====
    pam_a_lcc = _summary_val("PAM", "alignment", "LCC")
    pam_a_srcc = _summary_val("PAM", "alignment", "SRCC")
    pam_a_ktau = _summary_val("PAM", "alignment", "K-Tau")

    cmi_align_overall = _summary_val("CMI-Pref", "alignment", "ACC")
    cmi_text = _detail_acc("CMI-Pref", "alignment", "modality", "text")
    cmi_lyrics = _detail_acc("CMI-Pref", "alignment", "modality", "text+lyrics")
    cmi_audio = _detail_acc("CMI-Pref", "alignment", "modality", "text+ref")
    cmi_text_lyrics_ref = _detail_acc("CMI-Pref", "alignment", "modality", "text+lyrics+ref")
    cmi_align_all = cmi_text_lyrics_ref

    cmi_wo_audio = _safe_mean([cmi_text, cmi_lyrics])
    cmi_w_audio = _safe_mean([cmi_audio, cmi_text_lyrics_ref])

    # ===== Final compact block =====
    pam_mean_srcc = _safe_mean([pam_m_srcc, pam_a_srcc])
    cmi_mean_acc = _safe_mean([cmi_align_overall, cmi_music_acc])

    musicality_line = " & ".join([
        training_variant,
        _fmt4(pam_m_lcc),
        _fmt4(pam_m_srcc),
        _fmt4(pam_m_ktau),
        _fmt4(me_lcc),
        _fmt4(me_srcc),
        _fmt4(me_ktau),
        _fmt_pct(ma_acc),
        _fmt_pct(cmi_music_acc),
    ]) + r" \\" 

    alignment_line = " & ".join([
        training_variant,
        _fmt4(pam_a_lcc),
        _fmt4(pam_a_srcc),
        _fmt4(pam_a_ktau),
        _fmt_pct(cmi_text),
        _fmt_pct(cmi_lyrics),
        _fmt_pct(cmi_audio),
        _fmt_pct(cmi_align_all),
        _fmt_pct(cmi_wo_audio),
        _fmt_pct(cmi_w_audio),
    ]) + r" \\" 

    final_line = " & ".join([
        training_variant,
        _fmt4(pam_mean_srcc),
        _fmt4(me_srcc),
        _fmt_pct(ma_acc),
        _fmt_pct(cmi_mean_acc),
    ]) + r" \\" 

    lines = [
        "% Musicality: Method&Model, PAM(LCC,SRCC,K-Tau), MusicEval(LCC,SRCC,K-Tau), MusicArena(ACC), CMI-Pref(ACC)",
        musicality_line,
        "% Alignment: Method&Model, PAM(LCC,SRCC,K-Tau), CMI(text,lyrics,audio,all=text+lyrics+ref,w/o audio,w/ audio)",
        alignment_line,
        "% Final: Training variant, PAM Mean SRCC, MusicEval SRCC, Music Arena Acc, CMI-Pref Mean Acc",
        final_line,
        "% CMI-Pref Musicality: variant, Total, Conf<3, Conf=3, Conf>3, Instru(no lyrics), Vocal(lyrics)",
        cmi_musicality_line,
        "% CMI-Pref Text-Music Alignment (no ref): variant, Total, Conf<3, Conf=3, Conf>3, Instru(text), Vocal(text+lyrics)",
        cmi_tma_line,
        "% CMI-Pref Audio-Music Alignment (ref): variant, Total, Conf<3, Conf=3, Conf>3, Instru(text+ref), Vocal(text+lyrics+ref)",
        cmi_ama_line,
    ]

    temp_txt = out_dir / cfg.get("outputs", {}).get("temp_ordered_latex", "latex.txt")
    with open(temp_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return str(temp_txt)


def evaluate_from_rows(rows: List[Dict[str, Any]], args: EvaluateArgs) -> Dict[str, Any]:
    out_dir = Path(args.output_dir_abs)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_yaml_abs:
        args.eval_yaml_abs = str(Path(__file__).parent / "config" / "eval_benchmark.yaml")

    cfg = _load_cfg(args.eval_yaml_abs)
    df = pd.DataFrame(rows)

    df["benchmark_dataset"] = df.apply(_infer_benchmark_dataset, axis=1)

    split_target = str(cfg["filter"].get("split", "test"))
    if "split" in df.columns:
        df = df[df["split"].astype(str) == split_target].copy()

    df["month"] = df.apply(_extract_month, axis=1)
    df["modality_group"] = df.apply(_infer_modality, axis=1)

    include_lcc = bool(cfg["metrics"].get("include_lcc", False))
    include_k_tau = bool(cfg["metrics"].get("include_k_tau", False))

    summary_rows: List[Dict[str, Any]] = _compute_summary_rows(
        df=df,
        include_lcc=include_lcc,
        include_k_tau=include_k_tau,
    )
    detail_rows: List[Dict[str, Any]] = []

    ma = df[df["benchmark_dataset"] == "MusicArena"] if "benchmark_dataset" in df.columns else df.iloc[0:0]
    if not ma.empty:
        if bool(cfg["groups"].get("musicarena_by_month", False)):
            for month, g in ma.groupby("month"):
                g_acc = _acc(g, "preference", "predicted_preference")
                detail_rows.append({
                    "dataset": "MusicArena",
                    "task": "preference",
                    "group_key": "month",
                    "group_value": month,
                    "n": g_acc["n"],
                    "ACC": g_acc["acc"],
                })

    cmi = df[df["benchmark_dataset"] == "CMI-Pref"] if "benchmark_dataset" in df.columns else df.iloc[0:0]
    if not cmi.empty:
        if bool(cfg["groups"].get("cmi_by_modality", False)):
            for modality, g in cmi.groupby("modality_group"):
                for task_name, gt_col, pred_col in [
                    ("alignment", "preference-alignment", "predicted_preference-alignment"),
                    ("musicality", "preference-musicality", "predicted_preference-musicality"),
                ]:
                    g_acc = _acc(g, gt_col, pred_col)
                    detail_rows.append({
                        "dataset": "CMI-Pref",
                        "task": task_name,
                        "group_key": "modality",
                        "group_value": modality,
                        "n": g_acc["n"],
                        "ACC": g_acc["acc"],
                    })

        if bool(cfg["groups"].get("cmi_by_confidence", False)):
            _NO_REF_MODALITIES = {"text", "text+lyrics"}
            _REF_MODALITIES = {"text+ref", "text+lyrics+ref"}

            for task_name, gt_col, pred_col, confidence_col in [
                ("alignment", "preference-alignment", "predicted_preference-alignment", "confidence_preference-alignment"),
                ("musicality", "preference-musicality", "predicted_preference-musicality", "confidence_preference-musicality"),
            ]:
                if confidence_col not in cmi.columns:
                    continue

                cmi_conf_task = cmi.copy()
                cmi_conf_task["confidence_group"] = cmi_conf_task[confidence_col].apply(_confidence_group)
                cmi_conf_task["ref_category"] = cmi_conf_task["modality_group"].apply(
                    lambda m: "ref" if str(m) in _REF_MODALITIES else "no_ref"
                )
                cmi_conf_task = cmi_conf_task[cmi_conf_task["confidence_group"].notna()].copy()

                for (conf_group, ref_cat), g in cmi_conf_task.groupby(
                    ["confidence_group", "ref_category"], sort=True
                ):
                    g_acc = _acc(g, gt_col, pred_col)
                    detail_rows.append({
                        "dataset": "CMI-Pref",
                        "task": task_name,
                        "group_key": "confidence",
                        "group_value": conf_group,
                        "ref_category": ref_cat,
                        "n": g_acc["n"],
                        "ACC": g_acc["acc"],
                    })

                # Also emit combined (all) rows — used for musicality latex column
                for conf_group, g in cmi_conf_task.groupby("confidence_group", sort=True):
                    g_acc = _acc(g, gt_col, pred_col)
                    detail_rows.append({
                        "dataset": "CMI-Pref",
                        "task": task_name,
                        "group_key": "confidence",
                        "group_value": conf_group,
                        "ref_category": "all",
                        "n": g_acc["n"],
                        "ACC": g_acc["acc"],
                    })

        # ref_subset totals (no_ref / ref) — for text-music and audio-music total columns
        _NO_REF_MODS = {"text", "text+lyrics"}
        _REF_MODS = {"text+ref", "text+lyrics+ref"}
        for ref_cat, mods in [("no_ref", _NO_REF_MODS), ("ref", _REF_MODS)]:
            g = cmi[cmi["modality_group"].isin(mods)]
            for task_name, gt_col, pred_col in [
                ("alignment", "preference-alignment", "predicted_preference-alignment"),
                ("musicality", "preference-musicality", "predicted_preference-musicality"),
            ]:
                g_acc = _acc(g, gt_col, pred_col)
                detail_rows.append({
                    "dataset": "CMI-Pref",
                    "task": task_name,
                    "group_key": "ref_subset",
                    "group_value": ref_cat,
                    "n": g_acc["n"],
                    "ACC": g_acc["acc"],
                })

        # modality_type: instru (no lyrics) / vocal (has lyrics) across ref and no-ref
        for modality_type, mods in [
            ("instru", {"text", "text+ref"}),
            ("vocal", {"text+lyrics", "text+lyrics+ref"}),
        ]:
            g = cmi[cmi["modality_group"].isin(mods)]
            for task_name, gt_col, pred_col in [
                ("alignment", "preference-alignment", "predicted_preference-alignment"),
                ("musicality", "preference-musicality", "predicted_preference-musicality"),
            ]:
                g_acc = _acc(g, gt_col, pred_col)
                detail_rows.append({
                    "dataset": "CMI-Pref",
                    "task": task_name,
                    "group_key": "modality_type",
                    "group_value": modality_type,
                    "n": g_acc["n"],
                    "ACC": g_acc["acc"],
                })

    if bool(cfg["groups"].get("by_duration", False)):
        bins_raw = cfg["groups"].get("duration_bin", [])
        bins: List[float] = []
        for b in bins_raw:
            val = pd.to_numeric(b, errors="coerce")
            if not pd.isna(val):
                bins.append(float(val))

        if len(bins) >= 2:
            df_with_duration = df.copy()
            df_with_duration["duration"] = df_with_duration.apply(_row_duration_seconds, axis=1)
            df_with_duration["duration_group"] = df_with_duration["duration"].apply(lambda x: _duration_bin_label(x, bins))
            df_with_duration = df_with_duration[df_with_duration["duration_group"].notna()].copy()

            for duration_group, g in df_with_duration.groupby("duration_group"):
                g_summary_rows = _compute_summary_rows(
                    df=g,
                    include_lcc=False,
                    include_k_tau=False,
                )
                for row in g_summary_rows:
                    detail_row: Dict[str, Any] = {
                        "dataset": row.get("dataset"),
                        "task": row.get("task"),
                        "group_key": "duration_bin",
                        "group_value": duration_group,
                        "n": row.get("n"),
                    }
                    if "SRCC" in row:
                        detail_row["SRCC"] = row.get("SRCC")
                    if "ACC" in row:
                        detail_row["ACC"] = row.get("ACC")
                    detail_rows.append(detail_row)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_path = out_dir / cfg["outputs"].get("summary_csv", "metrics_summary.csv")
    detail_path = out_dir / cfg["outputs"].get("detail_csv", "metrics_detail.csv")
    json_path = out_dir / cfg["outputs"].get("json", "metrics.json")

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    payload = {
        "config": cfg,
        "summary": summary_rows,
        "detail": detail_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved summary: {summary_path}")
    logger.info(f"Saved detail: {detail_path}")
    logger.info(f"Saved json: {json_path}")

    temp_ordered_latex = _build_temp_ordered_latex(
        summary_rows=summary_rows,
        detail_rows=detail_rows,
        cfg=cfg,
        out_dir=out_dir,
    )
    logger.info(f"Saved temp ordered latex: {temp_ordered_latex}")

    print("summary:")
    print(summary_df)
    print("detail:")
    print(detail_df)
    return {
        "summary_csv": str(summary_path),
        "detail_csv": str(detail_path),
        "json": str(json_path),
        "temp_ordered_latex": str(temp_ordered_latex),
        "summary_rows": len(summary_rows),
        "detail_rows": len(detail_rows),
    }


def evaluate_results_main(args: EvaluateArgs) -> Dict[str, Any]:
    if not args.results_jsonl_abs:
        raise ValueError("results_jsonl_abs is required for file-based evaluation")
    rows = _load_jsonl(args.results_jsonl_abs)
    return evaluate_from_rows(rows=rows, args=args)


def main_func(args: EvaluateArgs) -> Dict[str, Any]:
    return evaluate_results_main(args)


def _parse_args() -> EvaluateArgs:
    parser = argparse.ArgumentParser(description="Benchmark evaluation prototype")
    parser.add_argument("--results_jsonl_abs", "-r", required=True)
    parser.add_argument("--output_dir_abs", default=None)
    parser.add_argument("--eval_yaml_abs", default=None)
    ns = parser.parse_args()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not ns.output_dir_abs:
        ns.output_dir_abs = str(Path(ns.results_jsonl_abs).parent / "evaluation_output" / timestamp)
    if not ns.eval_yaml_abs:
        ns.eval_yaml_abs = str(Path(__file__).parent / "config" / "eval_benchmark.yaml")
    return EvaluateArgs(**vars(ns))


def main() -> None:
    args = _parse_args()
    main_func(args)


if __name__ == "__main__":
    main()
