# 训练与评估完整流程

本文档描述当前项目在“不做实机、不做真实数据微调”的情况下，从已生成数据到模型训练、闭环评估、结果汇总的完整步骤。

当前默认数据集是 Potsdam，`train_data_generate.py` 的默认 `IMG_SIZE = 336`，因此单卡 4090 优先使用 `Profile 336`。

## 0. 目录约定

假设当前工作目录是：

```powershell
C:\Users\Lytem\Downloads\Programming\.vscode\airsim\AirNav\code
```

假设 LLaMA-Factory 在：

```powershell
..\LLaMA-Factory
```

假设数据生成输出目录是：

```powershell
..\data\potsdam_out
```

如果你的实际输出目录不同，把后续命令里的 `..\data\potsdam_out` 换成真实路径。

生成目录里至少应有：

```text
train.jsonl
eval.jsonl
train_images/
eval_images/
image_ids_train.txt
image_ids_val.txt
train_manifest.jsonl
eval_manifest.jsonl
dataset_info.json
```

## 1. 数据生成后先校验

先运行校验脚本，确认 ShareGPT 格式、图片路径、`<image>` 数量、PID 划分、防泄漏、场景比例和高度比例都正常：

```powershell
python validate_generated_data.py --output_dir ..\data\potsdam_out
```

通过时会看到类似：

```text
Validation passed with 0 warning(s).
```

有 warning 不一定不能训练，但要看内容：

- `ratio drift`：说明场景或高度比例偏离 PLAN 目标，样本少时常见；正式数据建议尽量控制在 3% 左右。
- `dataset_info.json not found`：说明数据生成时没同步注册，但后续 `prepare_llamafactory_dataset.py` 仍会给 LLaMA-Factory 注册。
- `missing image` / `PID leakage` / `<image> count` 是硬错误，必须先修。

说明：`validate_generated_data.py` 已兼容当前 Potsdam tile id，例如 `4_11_traj000000_step00.jpg`；也兼容 DOTA 的 `P0001_traj...`。

## 2. 准备 LLaMA-Factory 数据注册

训练前要把当前项目生成的 `train.jsonl/eval.jsonl` 注册到 LLaMA-Factory。运行：

```powershell
python training\prepare_llamafactory_dataset.py `
  --generated_dir ..\data\potsdam_out `
  --llamafactory_dir ..\LLaMA-Factory
```

Linux/macOS 对应命令：

```bash
python training/prepare_llamafactory_dataset.py \
  --generated_dir ../data/potsdam_out \
  --llamafactory_dir ../LLaMA-Factory
```

这个脚本会写入：

```text
..\LLaMA-Factory\data\uav_nav\train.jsonl
..\LLaMA-Factory\data\uav_nav\eval.jsonl
..\LLaMA-Factory\data\dataset_info.json
```

它还会把 JSONL 里的图片路径改成绝对路径，避免切换到 LLaMA-Factory 目录后找不到图片。

不要在 LLaMA-Factory YAML 里使用 `val_size`。本项目已经按整张图划分 `train/eval`，训练配置必须使用：

```yaml
dataset: uav_full
eval_dataset: uav_eval
```

## 3. 冒烟训练

先跑 10 step 冒烟测试，排除路径、依赖、显存、数据注册问题：

```powershell
.\training\run_train.ps1 `
  -GeneratedDir ..\data\potsdam_out `
  -LlamaFactoryDir ..\LLaMA-Factory `
  -Profile smoke `
  -RunName smoke_check
```

Linux/macOS 对应命令：

```bash
bash training/run_train.sh ../data/potsdam_out ../LLaMA-Factory smoke smoke_check
```

冒烟配置文件是：

```text
training\qwen2_5vl_lora_sft_smoke.yaml
```

输出目录示例：

```text
..\LLaMA-Factory\saves\qwen2_5vl-7b\lora\uav_sft_smoke\smoke_check
```

如果 smoke 阶段 OOM，先确认没有误用 448 配置；再把 smoke YAML 里的 `cutoff_len` 从 `3072` 降到 `2048`。

## 4. 正式训练

### 单卡 4090 / IMG_SIZE=336

*注意：如果使用V100S，应该将配置文件的bf16修改为fp16，infer_dtype从bfloat16修改为float16*

当前默认推荐：

```powershell
.\training\run_train.ps1 `
  -GeneratedDir ..\data\potsdam_out `
  -LlamaFactoryDir ..\LLaMA-Factory `
  -Profile 336 `
  -RunName potsdam_2575_v1
```

Linux/macOS 对应命令：

```bash
CUDA_VISIBLE_DEVICES=0 bash training/run_train.sh ../data/potsdam_out ../LLaMA-Factory 336 run_decision decision_action
bash training/run_train.sh ../data/potsdam_out ../LLaMA-Factory 336 potsdam_2575_v1 decision_action /tmp/AirNav/LLaMA-Factory/saves/qwen2_5vl-7b/lora/uav_sft_336/potsdam_2575_v1/checkpoint-1200
```

对应配置：

```text
training\qwen2_5vl_lora_sft_336.yaml
```

输出目录示例：

```text
..\LLaMA-Factory\saves\qwen2_5vl-7b\lora\uav_sft_336\potsdam_2575_v1
```

### 双卡或更大显存 / IMG_SIZE=448

只有当你把 `train_data_generate.py` 里的：

```python
IMG_SIZE = 448
```

并重新生成了训练数据后，才使用：

```powershell
.\training\run_train.ps1 `
  -GeneratedDir ..\data\potsdam_out `
  -LlamaFactoryDir ..\LLaMA-Factory `
  -Profile 448 `
  -RunName potsdam_448_v1
```

Linux/macOS 对应命令：

```bash
bash training/run_train.sh ../data/potsdam_out ../LLaMA-Factory 448 potsdam_448_v1
```

对应输出示例：

```text
..\LLaMA-Factory\saves\qwen2_5vl-7b\lora\uav_sft_448\potsdam_448_v1
```

注意：如果数据是 336 生成的，却用 448 配置训练，LLaMA-Factory 会把图片再次放大到 448，浪费显存和时间。

## 5. 查看训练是否正常

训练期间重点看：

- loss 是否下降；
- eval loss 是否明显高于 train loss；
- 是否频繁 OOM；
- 保存目录下是否出现 checkpoint。

训练输出通常包括：

```text
adapter_config.json
adapter_model.safetensors
trainer_state.json
training_args.bin
```

最终评估使用完整输出目录，不要只拿某一个文件。

## 6. 启动 zero-shot API 服务

`eval.py` 的 `zero_shot/finetuned` 策略通过 OpenAI-compatible API 调模型。先在一个单独 PowerShell 窗口启动 base 模型服务：

```powershell
cd ..\LLaMA-Factory
$env:API_PORT="8000"
llamafactory-cli api ..\code\training\qwen2_5vl_zero_shot_api.yaml
```

```bash
API_PORT=8000 llamafactory-cli api ../code/training/qwen2_5vl_zero_shot_api.yaml
```

服务起来后，回到 `code` 目录，跑 zero-shot 闭环评估。

## 7. 跑 zero-shot 评估

Potsdam 当前推荐评估方式是：在 `train` split 上读取生成目录中的 `image_ids_val.txt` 做 held-out 闭环评估。`eval.py` 会同时读取 `generation_config.json`，默认对齐数据生成时的 `landmark/max_distance/min_start_distance` 参数以及场景集合。

Potsdam zero-shot 评估示例：

```powershell
python eval.py `
  --data_root ..\data\Potsdam `
  --generated_dir ..\data\potsdam_out `
  --dataset potsdam `
  --split train `
  --num_episodes 50 `
  --policies random,greedy,zero_shot `
  --zero_shot_base_url http://localhost:8000/v1 `
  --zero_shot_model Qwen2.5-VL-7B `
  --output_dir eval_zero_shot
```

Linux/macOS 对应命令：

```bash
python eval.py \
  --data_root ../data/Potsdam \
  --generated_dir ../data/potsdam_out \
  --dataset potsdam \
  --split train \
  --num_episodes 50 \
  --policies random,greedy,zero_shot \
  --zero_shot_base_url http://localhost:8000/v1 \
  --zero_shot_model Qwen2.5-VL-7B \
  --output_dir eval_zero_shot
```

输出文件：

```text
eval_zero_shot\eval_results.jsonl
eval_zero_shot\eval_summary.json
eval_zero_shot\eval_summary.csv
```

如果你有独立 OOD `val` split，可改成：

```powershell
--split val
```

这种情况下如果不想使用 `image_ids_val.txt`，就不要传 `--generated_dir`，或者显式传 `--eval_pid_list` 指向你自己的 held-out 列表。

最终报告建议至少给：

- Random
- Greedy
- Zero-shot Qwen2.5-VL-7B
- Fine-tuned Qwen2.5-VL-7B-LoRA

## 8. 启动 finetuned API 服务

先停止 zero-shot API 服务，然后启动 LoRA 模型服务。

单卡 336 训练结果：

```powershell
cd ..\LLaMA-Factory
$env:API_PORT="8000"
llamafactory-cli api ..\code\training\qwen2_5vl_lora_api_336.yaml
```

```bash
API_PORT=8000 llamafactory-cli api ../code/training/qwen2_5vl_lora_api_336.yaml
```

如果你训练的是 448 配置：

```powershell
cd ..\LLaMA-Factory
$env:API_PORT="8000"
llamafactory-cli api ..\code\training\qwen2_5vl_lora_api_448.yaml
```

*注意：需要修改api的yaml的adapter_name_or_path为checkpoint的实际位置*
这些 API 配置里的 `adapter_name_or_path` 默认是相对 LLaMA-Factory 根目录的基路径：

```text
saves/qwen2_5vl-7b/lora/uav_sft_336
saves/qwen2_5vl-7b/lora/uav_sft_448
```

如果你使用了 `-RunName`，要把 API YAML 里的 `adapter_name_or_path` 改成具体 run 子目录，例如：

```text
saves/qwen2_5vl-7b/lora/uav_sft_336/potsdam_2575_v1
```

## 9. 跑 finetuned 闭环评估

启动 LoRA API 后，回到 `code` 目录：

```powershell
python eval.py `
  --data_root ..\data\Potsdam `
  --generated_dir ..\data\potsdam_out `
  --dataset potsdam `
  --split train `
  --num_episodes 50 `
  --policies random,greedy,finetuned `
  --finetuned_base_url http://localhost:8000/v1 `
  --finetuned_model Qwen2.5-VL-7B-UAV `
  --output_dir eval_finetuned
```

Linux/macOS 对应命令：

```bash
python eval.py \
  --data_root ../data/Potsdam \
  --generated_dir ../data/potsdam_out \
  --dataset potsdam \
  --split train \
  --num_episodes 100 \
  --policies random,greedy,finetuned \
  --finetuned_base_url http://localhost:8000/v1 \
  --finetuned_model Qwen2.5-VL-7B-UAV \
  --output_dir eval_finetuned \
  --stop_grace_steps 3 \
  --assistant_format decision_action
```

如果要跟 zero-shot 使用完全相同的 episode，保持相同：

```text
--seed
--data_root
--generated_dir
--dataset
--split
--num_episodes
--max_images
```

默认 `seed=42`，不显式传也会固定。

## 10. 推荐最终评估命令

先跑一个小规模版本确认链路：

```powershell
python eval.py `
  --data_root ..\data\Potsdam `
  --generated_dir ..\data\potsdam_out `
  --dataset potsdam `
  --split train `
  --num_episodes 5 `
  --max_steps 20 `
  --policies random,greedy,finetuned `
  --finetuned_base_url http://localhost:8000/v1 `
  --finetuned_model Qwen2.5-VL-7B-UAV `
  --output_dir eval_finetuned_smoke
```

Linux/macOS 对应命令：

```bash
python eval.py \
  --data_root ../data/Potsdam \
  --generated_dir ../data/potsdam_out \
  --dataset potsdam \
  --split train \
  --num_episodes 5 \
  --max_steps 20 \
  --policies random,greedy,finetuned \
  --finetuned_base_url http://localhost:8000/v1 \
  --finetuned_model Qwen2.5-VL-7B-UAV \
  --output_dir eval_finetuned_smoke
```

再跑正式结果：

```powershell
python eval.py `
  --data_root ..\data\Potsdam `
  --generated_dir ..\data\potsdam_out `
  --dataset potsdam `
  --split train `
  --num_episodes 100 `
  --max_steps 20 `
  --policies random,greedy,finetuned `
  --finetuned_base_url http://localhost:8000/v1 `
  --finetuned_model Qwen2.5-VL-7B-UAV `
  --output_dir eval_finetuned_final
```

Linux/macOS 对应命令：

```bash
python eval.py \
  --data_root ../data/Potsdam \
  --generated_dir ../data/potsdam_out \
  --dataset potsdam \
  --split train \
  --num_episodes 100 \
  --max_steps 20 \
  --policies random,greedy,finetuned \
  --finetuned_base_url http://localhost:8000/v1 \
  --finetuned_model Qwen2.5-VL-7B-UAV \
  --output_dir eval_finetuned_final
```

如果有 `val` split，可以补充一组 OOD 结果：

```powershell
--split val
```

主报告优先使用 `--generated_dir ..\data\potsdam_out --split train` 这条 held-out 结果；`--split val` 作为额外泛化对照。

## 11. 结果怎么看

核心文件是：

```text
eval_finetuned_final\eval_summary.csv
```

字段含义：

- `policy`: random / greedy / zero_shot / finetuned
- `height_m`: overall 或 25/50/75
- `n`: episode 数
- `sr`: Success Rate
- `spl`: Success weighted by Path Length
- `avg_steps`: 成功 episode 的平均步数

报告里建议使用这个表：

| Policy | SR | SPL | Avg Steps | SR@25m | SR@50m | SR@75m |
| --- | --- | --- | --- | --- | --- | --- |
| Random | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 |
| Greedy | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 |
| Zero-shot | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 |
| Fine-tuned | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 | 从 CSV 填 |

## 12. 常见问题

### validate 报图片名不匹配

确认你已经使用当前版本的 `validate_generated_data.py`。旧版只接受 `P0001_traj...`，不接受 Potsdam 的 `4_11_traj...`。

### LLaMA-Factory 训练时找不到图片

重新运行：

```powershell
python training\prepare_llamafactory_dataset.py `
  --generated_dir ..\data\potsdam_out `
  --llamafactory_dir ..\LLaMA-Factory
```

不要直接把原始 JSONL 复制到 LLaMA-Factory 后训练，因为相对路径可能失效。

### API 服务启动后 eval.py 连不上

检查：

- API 窗口是否仍在运行；
- 端口是否是 `8000`；
- `eval.py` 是否传了 `http://localhost:8000/v1`；
- 防火墙或代理是否拦截本地端口。

### 单卡 OOM

优先使用 `Profile 336`。仍然 OOM 时：

1. 把 `training\qwen2_5vl_lora_sft_336.yaml` 的 `cutoff_len` 从 `3072` 降到 `2048`。
2. 保持 `per_device_train_batch_size: 1`。
3. 不要把 `image_max_pixels` 改成 448 的 `200704`。

### zero-shot 和 finetuned 无法同时跑

当前 `eval.py` 对不同模型使用同一个 OpenAI API 入口。最稳妥流程是分两次跑：

1. 启 base API，跑 `zero_shot`，保存 `eval_zero_shot`。
2. 启 LoRA API，跑 `finetuned`，保存 `eval_finetuned`。

只要 `seed` 和 episode 参数一致，两次结果可以直接对比。

## 13. 完成标准

不包含实机和真实数据微调时，项目完成标准是：

1. `validate_generated_data.py --output_dir <generated_dir>` 通过。
2. `Profile smoke` 训练通过。
3. `Profile 336` 或 `Profile 448` 正式训练完成，产出 LoRA adapter。
4. `random/greedy/zero_shot/finetuned` 至少各跑一轮闭环评估。
5. `eval_summary.csv` 中有 overall 和按高度分桶的 SR/SPL/AvgSteps。

做到以上，仿真数据 + 单阶段 SFT + 闭环 SR/SPL 评估这一核心交付就算闭环完成。
