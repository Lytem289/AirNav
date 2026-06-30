# UAV VLN training with LLaMA-Factory

这个目录补齐训练阶段需要的配置和脚本。数据生成脚本已经会产出：

- `train.jsonl`
- `eval.jsonl`
- `train_images/`
- `eval_images/`
- `image_ids_train.txt`
- `image_ids_val.txt`
- `train_manifest.jsonl`
- `eval_manifest.jsonl`

训练阶段只做三件事：

1. 把生成目录注册到 LLaMA-Factory。
2. 用 Qwen2.5-VL-7B-Instruct 做单阶段 LoRA SFT。
3. 保存 LoRA adapter，后续给 `eval.py` 或推理服务加载。

## 文件说明

- `qwen2_5vl_lora_sft_smoke.yaml`: 10 step 冒烟测试，先跑这个排查路径、图片和显存问题。
- `qwen2_5vl_lora_sft_336.yaml`: 单卡 4090 24GB 推荐配置，对应 `train_data_generate.py` 里的 `IMG_SIZE = 336`。
- `qwen2_5vl_lora_sft_448.yaml`: 双卡 4090 或更大显存配置，对应 `IMG_SIZE = 448`。
- `prepare_llamafactory_dataset.py`: 把生成的 JSONL 拷贝到 LLaMA-Factory，并注册 `uav_full/uav_eval`。
- `run_train.ps1`: Windows PowerShell 一键准备数据并训练。
- `run_train.sh`: Linux/macOS Bash 一键准备数据并训练。

## 放到哪里

这些文件保留在当前项目的 `code/training/` 目录即可。运行脚本会自动把数据注册到：

```text
../LLaMA-Factory/data/dataset_info.json
../LLaMA-Factory/data/uav_nav/train.jsonl
../LLaMA-Factory/data/uav_nav/eval.jsonl
```

训练 YAML 不需要复制到 LLaMA-Factory；脚本会先生成一份仅覆盖 `output_dir` 的临时 YAML，再把这份单文件配置传给 `llamafactory-cli train`。这是为了兼容当前 LLaMA-Factory 对 `train <config.yaml>` 的解析方式。

## 训练前检查

1. 确认 LLaMA-Factory 已安装并能运行：

```powershell
cd ..\LLaMA-Factory
llamafactory-cli version
```

2. 确认你的生成目录里至少有：

```text
train.jsonl
eval.jsonl
train_images/
eval_images/
```

3. 如果你用默认 `train_data_generate.py`，当前 `IMG_SIZE = 336`，正式训练选 `336` 配置。只有把生成脚本里的 `IMG_SIZE` 改成 `448` 并重新生成数据后，才选 `448` 配置。

## Windows 运行

先跑冒烟测试：

```powershell
.\training\run_train.ps1 -GeneratedDir ..\data\potsdam_out -LlamaFactoryDir ..\LLaMA-Factory -Profile smoke -RunName smoke_check
```

冒烟测试通过后，单卡 4090 正式训练：

```powershell
.\training\run_train.ps1 -GeneratedDir ..\data\potsdam_out -LlamaFactoryDir ..\LLaMA-Factory -Profile 336 -RunName potsdam_2575_v1
```

双卡或更大显存，且数据是 `IMG_SIZE = 448` 重新生成的：

```powershell
.\training\run_train.ps1 -GeneratedDir ..\data\potsdam_out -LlamaFactoryDir ..\LLaMA-Factory -Profile 448 -RunName potsdam_448_v1
```

如果只想重新训练、不想重新准备数据：

```powershell
.\training\run_train.ps1 -GeneratedDir ..\data\potsdam_out -LlamaFactoryDir ..\LLaMA-Factory -Profile 336 -RunName retry_336_v2 -SkipPrepare
```

如果想直接把现有 JSONL 改成仅监督动作、不重跑轨迹生成：

```powershell
.\training\run_train.ps1 -GeneratedDir ..\data\potsdam_out -LlamaFactoryDir ..\LLaMA-Factory -Profile 336 -RunName retry_336_action_only -AssistantFormat action_only
```

## Linux/macOS 运行

```bash
bash training/run_train.sh ../data/potsdam_out ../LLaMA-Factory smoke smoke_check
bash training/run_train.sh ../data/potsdam_out ../LLaMA-Factory 336 potsdam_2575_v1
bash training/run_train.sh ../data/potsdam_out ../LLaMA-Factory 336 potsdam_2575_action_only action_only
bash training/run_train.sh ../data/potsdam_out ../LLaMA-Factory 336 potsdam_2575_action_only action_only saves/qwen2_5vl-7b/lora/uav_sft_336/potsdam_2575_action_only/checkpoint-1200
```

## 输出位置

训练输出在 LLaMA-Factory 目录下，并且每次训练默认都会放到一个独立的 `run_<profile>_<timestamp>` 子目录，避免覆盖旧实验。示例：

```text
../LLaMA-Factory/saves/qwen2_5vl-7b/lora/uav_sft_smoke/run_smoke_20260525_120000
../LLaMA-Factory/saves/qwen2_5vl-7b/lora/uav_sft_336/potsdam_2575_v1
../LLaMA-Factory/saves/qwen2_5vl-7b/lora/uav_sft_448/potsdam_448_v1
```

后续评估时加载具体的 run 子目录即可。

## 关键训练设定

- `stage: sft`
- `finetuning_type: lora`
- `dataset: uav_full`
- `eval_dataset: uav_eval`
- 不使用 `val_size`，避免同一张大图衍生样本随机切分导致泄漏。
- `freeze_vision_tower: true`
- `freeze_multi_modal_projector: false`
- LoRA 只挂 LLM 侧常见线性层：`q/k/v/o/gate/up/down_proj`
- `fp16: true`
- `gradient_checkpointing: true`

## Assistant 标签格式

- `thought_action`: 保留 `Thought:` 和 `Action:` 两部分，兼容旧数据。
- `action_only`: 只监督 `Action: (...)` 或 `Action: STOP`，适合当前闭环控制退化到固定话术时做对照实验。
- `decision_action`: 监督 `Decision: MOVE/STOP` + `Action: ...` 两行输出，把是否停止拆成显式二分类决策。
- `prepare_llamafactory_dataset.py` 默认 `--assistant_format auto`：优先读取 `generation_config.json` 里的 `assistant_format`，没有则回退到 `thought_action`。

## 常见问题

如果提示图片找不到，重新执行 `run_train.ps1` 且不要加 `-SkipPrepare`。准备脚本会把 JSONL 里的图片路径重写为绝对路径。

如果单卡 4090 OOM，先确认你跑的是 `Profile 336`，并且 `train_data_generate.py` 的 `IMG_SIZE = 336`。仍然 OOM 时，把 `cutoff_len` 降到 `2048`，或把 `gradient_accumulation_steps` 保持不变但不要增大 batch size。

如果你改成 `IMG_SIZE = 448`，必须重新生成训练数据，并使用 `Profile 448`，否则 LLaMA-Factory 会额外重采样图片，浪费显存和时间。
