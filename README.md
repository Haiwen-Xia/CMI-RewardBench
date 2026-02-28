# CMI-RewardBench

[![Arxiv Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/pdf/2505.10793)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This repository provides a unified benchmark and evaluation toolkit for **music reward modeling** under **Compositional Multimodal Instruction (CMI)**, where generated music can be conditioned on **text descriptions**, **lyrics**, and **reference audio prompts**.

### ✨ Key Features

- **CMI-RewardBench**: A unified benchmark covering **musicality**, **text–music alignment**, and **compositional instruction alignment**.
- **Evaluation Toolkit**: Everything you need to reproduce benchmark results and compute correlations with human judgments.
- **Inference Baseline**: A strong, unified, reusable inference interface to run our pre-trained **CMI Reward Models (CMI-RMs)**.
- **Data Assets**: Access to CMI-Pref-Pseudo, CMI-Pref, and other metadata seamlessly.

---

## 🗂️ Data Resources

| Resource | Type | Link |
| :--- | :--- | :--- |
| **CMI-Pref** | Human-annotated Dataset | [🤗 Hugging Face Dataset](https://huggingface.co/datasets/HaiwenXia/cmi-pref) |
| **CMI-Pref-Pseudo** | Pseudo-labeled Dataset | [🤗 Hugging Face Dataset](https://huggingface.co/datasets/HaiwenXia/cmi-pref-pseudo) |
| **CMI-RM** | Trained Reward Model | [🤗 Hugging Face Model](https://huggingface.co/HaiwenXia/CMI-RM) |
| **CMI-RewardBench** | Metadata JSONL | data/all_test.jsonl |

---

## 🚀 Quick Start

### 1. Installation

Install the standard packages for data downloading and evaluation:
```bash
pip install -r requirements.txt
```

*(Optional)* If you intend to use our pre-trained CMI-RM model as a strong baseline, install the baseline inference dependencies:
```bash
pip install -r baselines/requirements.txt
```

### 2. Prepare Data

Download the PAM-Music, MusicEval, Music Arena datasets, and our CMI-Pref evaluation dataset, placing them directly into the `data/` directory:
```bash
chmod +x download_data.sh
./download_data.sh
```

### 3. Run Benchmark with CMI-RM Baseline

We host our open-source model checkpoints on Hugging Face. We recommend using `huggingface-cli` for faster and more stable downloads to the `baselines/model` directory:
```bash
# Install huggingface_hub if you haven't already
pip install huggingface_hub

# Download the pre-trained checkpoint and configs
huggingface-cli download HaiwenXia/CMI-RM --local-dir baselines/model
```

Once the model is downloaded, simply start the inference benchmark! The system automatically utilizes the built-in adapter:
```bash
python inference_benchmark.py \
  -c baselines/model/model.safetensors \
  --dataset_jsonl data/all_test.jsonl \
  --dataset_root data/ \
  --device cuda:0 \
```

> **Note:** The script located at `baselines/inference.py` contains a general-purpose inference API decoupled from the benchmark. It is heavily recommended for users looking to use CMI-RMs in separate workflows!

---

## 🧩 Evaluating Your Own Model

We created a hassle-free, declarative architecture for testing custom baseline models. Just inherit our `BenchmarkModelABC` abstract base class and plug it dynamically into the command-line flags without modifying any core pipeline logic.

### Writing the Interface

Use `custom_models/sample_model.py` as the adapter entrypoint and implement a class inheriting `BenchmarkModelABC`.

The adapter contract is intentionally minimal:
- `sr` property (audio sample rate)
- `score_batch(...) -> np.ndarray` with shape `[batch_size, 2]`
- column `0` = alignment, column `1` = musicality

Recommended project layout:
- Put benchmark-facing adapter code in `custom_models/`
- Put model dependencies / third-party model source under `models/`
- Import your internal model implementation from `models/` inside the `custom_models/` adapter.


### Running the Custom Model

Initiate the benchmark pipeline by passing your Python path signature using `--model_class_path`.

Behavior details:
- `--model_class_path` has higher priority than `--model`; once provided, your class is used directly.
- You can also define your own `--model` choice in `inference_benchmark.py` as a shortcut to `--model_class_path`.
- By default, `checkpoint` and `device` are injected from CLI (`-c/--checkpoint`, `--device`).
- Extra constructor args can be passed via `--model_init_kwargs_json`.
- Extra runtime args for `score_batch(..., **kwargs)` can be passed via `--model_score_kwargs_json`.

```bash
python inference_benchmark.py \
  -c /path/to/my_weights.pt \
  --device cuda:0 \
  --dataset_jsonl data/all_test.jsonl \
  --dataset_root data/ \
  --model_class_path custom_models.my_model:CustomRewardModel \
  --model_init_kwargs_json '{"sr":24000}' \
  --model_score_kwargs_json '{"my_flag":true}'
```

*For more nuanced documentation on argument injection and workflow mapping, read the [Detailed Custom Model Interface Guide](model_interface.md).*

### Evaluation Outputs and `evaluate_results.py`

By default, `inference_benchmark.py` will call `evaluate_results.py` after inference (disable with `--no_run_evaluate`).

The evaluation stage writes structured metrics under the run directory:
- `metrics_summary.csv`: top-level benchmark metrics.
- `metrics_detail.csv`: grouped metrics with configurable granularity.
- `metrics.json`: JSON bundle of config + summary + detail.

Important evaluation behavior:
- MusicArena summary ACC is recomputed from `predicted_musicality` vs `predicted_musicality_b`.
- CMI-Pref uses pairwise preference labels (`predicted_preference-alignment`, `predicted_preference-musicality`) for ACC.

You can control output granularity in `config/eval_benchmark.yaml`:

---

## 🏆 Selected Competitive Results

### Musicality Evaluation

| Method/Model | PAM SRCC | MusicEval SRCC | Music Arena ACC | CMI-Pref ACC |
|---|---:|---:|---:|---:|
| **Gemini3-pro*** | 0.5967 | 0.6018 | 68.85% | 65.80% |
| **SongEval-RM** | **0.6977** | 0.6949 | **73.88%** | 72.40% |
| **CMI-RM (Ours)** | 0.6606 | **0.8266** | 73.43% | **78.20%** |

### Alignment Evaluation 

*CMI-Pref subcategory keys: **T** = Text–Music, **L** = Text+Lyrics–Music, **A** = Text+Audio–Music, **C** = Text+Lyrics+Audio*

| Method/Model | PAM SRCC | CMI-Pref (T) | CMI-Pref (L) | CMI-Pref (A) | CMI-Pref (C) |
|---|---:|---:|---:|---:|---:|
| **CLAP score (music)** | 0.2881 | 67.20% | **73.60%** | - | - |
| **Gemini3-pro*** | **0.5373** | 67.20% | 60.80% | 68.80% | 64.80% |
| **CMI-RM (Ours)** | 0.4321 | **67.60%** | 72.80% | **76.40%** | **79.20%** |



---

## 📂 Repository Structure

The core philosophy separates "benchmark runtime execution" from "evaluable baseline wrappers".

```text
CMI-RewardBench/
├── data/                    # Test-set jsons, raw audio arrays & script downloads
│   ├── all_test.jsonl       # Core test list details (text/lyrics/labels/audio)
│   └── all_train.jsonl      # Baseline tuning training subset (informational)
├── core/                    # Immutable ABC interface & default injection handler
├── baselines/               # Default CMI-RM model inference code and weights mapped space
├── config/                  # External structured YAML configs mapping evaluators
├── custom_models/           # User workspace sandbox for bridging new custom models
├── models/                  # Third-party model dependency folders
├── inference_benchmark.py   # Primary pipeline orchestrator entrypoint
└── model_interface.md       # Full parameter delegation API references
```
