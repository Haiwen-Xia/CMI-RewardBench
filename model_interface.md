# Benchmark Custom Model Interface

本说明介绍如何在当前仓库结构下接入自定义模型到 `inference_benchmark.py`。

## 0) 当前目录结构（关键部分）

- `core/`：benchmark 核心接口与默认适配器
  - `core/model_abc.py`：抽象类定义
  - `core/model_interface.py`：默认 `RewardModelAdapter`
- `custom_models/`：用户重载入口（你需要实现/修改的地方）
- `models/`：模型依赖代码（例如 `models/cmi-rm/src`）
- `baselines/`：官方 CMI-RM 推理接口（与 benchmark 解耦，可独立用于其他任务）

## 1) 抽象接口定义

接口定义在 `core/model_abc.py`。

必须实现：
- `sr` 属性（`int`）
- `score_batch(inputs, batch_size, max_dur, **kwargs) -> np.ndarray`

输入类型：`BenchmarkBatchInput`
- `audio`: 音频路径或 waveform tensor
- `text`: prompt 文本
- `lyrics`: 歌词文本
- `ref_audio`: 参考音频路径或 waveform tensor

输出格式：
- `np.ndarray`，shape 为 `[N, 2]`
- 第 0 列：alignment
- 第 1 列：musicality

## 2) 自定义模型放置位置

推荐放在：`custom_models/`

示例文件：`custom_models/sample_model.py`
示例类：`CMIRewardModelBaseline`

调用方式：
- `--model_class_path custom_models.sample_model:CMIRewardModelBaseline`

## 3) 自定义模型初始化行为

当提供 `--model_class_path` 时，benchmark 会执行：

`YourModelClass(**init_kwargs)`

并自动注入：
- `checkpoint`（来自 `--checkpoint/-c`）
- `device`（来自 `--device`）

可选扩展：
- `--model_init_kwargs_json`：传构造参数（例如 `{"sr":24000,"mode":"final","config":"/path/config.yaml"}`）
- `--model_score_kwargs_json`：透传给 `score_batch(..., **kwargs)`

## 4) 运行 benchmark

默认内置适配器（走 `core/model_interface.py`）：

```bash
python inference_benchmark.py \
  -c /path/to/model.safetensors \
  --dataset_jsonl data/all_test.jsonl \
  --dataset_root /path/to/dataset_root \
  --device cuda:0 \
  --reward_model_backend final
```

自定义模型：

```bash
python inference_benchmark.py \
  -c /path/to/model.safetensors \
  --device cuda:0 \
  --model_class_path custom_models.sample_model:CMIRewardModelBaseline \
  --model_init_kwargs_json '{"sr":24000,"mode":"final","config":"/path/to/config.yaml"}' \
  --dataset_jsonl data/all_test.jsonl \
  --dataset_root /path/to/dataset_root
```

## 5) `baselines/` 的定位

`baselines/` 不属于 benchmark 核心流程；它提供官方 CMI-RM 的通用推理接口，便于外部任务复用。

当前 baseline 约定：
- checkpoint 使用 `model.safetensors`
- 模型结构配置使用 `config.yaml`
- 推理模式支持 `mode=final` 和 `mode=standard`
