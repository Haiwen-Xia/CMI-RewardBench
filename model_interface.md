# Benchmark Custom Model Interface

This guide explains how to plug a custom model into `inference_benchmark.py` under the current repository structure.

## 0) Current directory layout (key parts)

- `core/`: benchmark core interfaces and default adapter
  - `core/model_abc.py`: abstract interface definitions
  - `core/model_interface.py`: default `RewardModelAdapter`
- `custom_models/`: user override entry points (where you implement custom models)
- `models/`: model dependency code (for example `models/cmi-rm/src`)
- `baselines/`: official CMI-RM inference interface (decoupled from benchmark, reusable for other tasks)

## 1) Abstract interface definition

The interface is defined in `core/model_abc.py`.

Required members:
- `sr` property (`int`)
- `score_batch(inputs, batch_size, max_dur, **kwargs) -> np.ndarray`

Input type: `BenchmarkBatchInput`
- `audio`: audio path or waveform tensor
- `text`: prompt text
- `lyrics`: lyrics text
- `ref_audio`: reference audio path or waveform tensor

Output format:
- `np.ndarray` with shape `[N, 2]`
- column 0: alignment
- column 1: musicality

## 2) Where to place custom models

Recommended location: `custom_models/`

Example file: `custom_models/sample_model.py`
Example class: `CMIRewardModelBaseline`

Usage:
- `--model_class_path custom_models.sample_model:CMIRewardModelBaseline`

## 3) Custom model initialization behavior

When `--model_class_path` is provided, benchmark executes:

`YourModelClass(**init_kwargs)`

and automatically injects:
- `checkpoint` (from `--checkpoint/-c`)
- `device` (from `--device`)

Optional extensions:
- `--model_init_kwargs_json`: pass constructor args (for example `{"sr":24000,"mode":"final","config":"/path/config.yaml"}`)
- `--model_score_kwargs_json`: forwarded to `score_batch(..., **kwargs)`

## 4) Run benchmark

Default built-in adapter (via `core/model_interface.py`):

```bash
python inference_benchmark.py \
  -c /path/to/model.safetensors \
  --dataset_jsonl data/all_test.jsonl \
  --dataset_root /path/to/dataset_root \
  --device cuda:0 \
  --reward_model_backend final
```

Custom model:

```bash
python inference_benchmark.py \
  -c /path/to/model.safetensors \
  --device cuda:0 \
  --model_class_path custom_models.sample_model:CMIRewardModelBaseline \
  --model_init_kwargs_json '{"sr":24000,"mode":"final","config":"/path/to/config.yaml"}' \
  --dataset_jsonl data/all_test.jsonl \
  --dataset_root /path/to/dataset_root
```

## 5) Role of `baselines/`

`baselines/` is not part of the benchmark core flow; it provides the official CMI-RM general inference API so users can reuse it outside benchmark tasks.

Current baseline conventions:
- checkpoint uses `model.safetensors`
- model structure config uses `config.yaml`
- inference supports both `mode=final` and `mode=standard`
