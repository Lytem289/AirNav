# Potsdam 重构说明

当前代码已从“只能跑 DOTA OBB 标注”重构为数据集适配式入口。默认数据集为 `Potsdam`，通过正射影像截图模拟无人机俯视飞行；DOTA 旧流程保留为 `--dataset dota` 兼容路径。

关键变化：
- `train_data_generate.py --dataset potsdam` 会扫描常见 Potsdam 目录：`train/images + train/labels`，或 ISPRS 原始 `2_Ortho_* / 3_Ortho_* / 5_Labels_*` 结构。
- Potsdam 没有 DOTA 的目标框标注，脚本会从语义 label mask 中按颜色提取连通域，把 `car/building/tree/low-vegetation/impervious-surface` 转成伪实例，后续轨迹生成、截图裁剪、ShareGPT 输出继续复用原有管线。
- Potsdam 使用真实 `GSD=0.05m/px`，不再依赖 DOTA 的 OBB 物理先验反推 GSD；评估端遇到 raster label 时也会使用该 GSD。
- 生成命令示例：

```bash
python train_data_generate.py --dataset potsdam --data_root ../data/potsdam --output_dir ../data/potsdam_out --num_trajectories 1000 --save_original_traj_vis
```

# 无人机视觉语言导航项目：实现指南与核心代码

本文档提供了基于 DOTA 数据集的无人机 VLN（Vision-Language Navigation）大创项目的完整落地架构、核心代码框架以及训练与测试流程。

## 1. 整体工程架构

> **【单位本质澄清】**
> 本项目所有"米"是基于 `FIXED_GSD=0.5 m/px` 的**评估端基准假设值**,DOTA 卫星图本身无可靠 GSD。
> 真正驱动模型学习的物理量是**像素**:
>   - `crop_size` (px) 决定视野大小
>   - `grid_pixel` (px) 决定 STOP 半径
>   - 起点像素距离 决定 episode 难度
> "米"仅作为内部 unit consistency 用,prompt 中不暴露(高度/grid_size/距离)。
> 模型输出无量纲 grid (x,y);实机部署时飞控用**真实 GSD**(由相机内参+飞行高度算出)换算成米——训练假设的 GSD 与实机真实 GSD 不需要严格对应。
> **训练端额外做 GSD 抖动数据增强**:`pixel_per_meter(gsd_jitter=True)` 每条轨迹独立抽样 `GSD ∈ [0.35, 0.7]`,让模型对绝对像素尺度更鲁棒;**评估端固定 GSD=0.5** (`pixel_per_meter(gsd_jitter=False)`) 保证 SR/SPL 复现性。详见 PLAN §4.1。
> **调用契约**:`ppm` 必须在轨迹/episode 入口调用一次,显式传给所有下游 (`sample_episode` / `compute_instruction_context_at_start` / `generate_trajectory` / `crop_uav_view`),严禁下游内部再次 `pixel_per_meter()`。

项目分为三个主要模块：
1.  **数据生成引擎 (Data Engine)**：解析 DOTA 标注，模拟无人机动力学（先转后走 + 噪声），裁剪并旋转高分辨率图像，生成包含完整思维链（CoT）的多模态 JSONL 数据集。
2.  **单阶段 SFT 训练管道 (Training Pipeline)**：基于 LLaMA-Factory 和 Qwen2.5-VL-7B 的 LoRA 微调。所有字段（2 张图 + 5 步历史动作 + 累计位移 + CoT + Action）一次性全量训练，**不再做** "Stage 1 mask 字段 + Stage 2 放开" 的两阶段课程——经评估，该设计会教会模型"忽略历史/累计位移"字段（反向 curriculum），与最终目标直接冲突。详见 PLAN §2.1 / §5.2。
3.  **闭环评估系统 (Evaluator)**：离线自回归推理循环，模拟真实的飞行过程，统计成功率 (SR) 和路径效率 (SPL)。

## 2. 核心数据生成脚本 (`generate_data.py`)

此脚本实现了从 DOTA 数据集到多模态 JSONL 数据的全自动生成。

> **起点分布说明**：上述函数仅生成单条轨迹，批量生成时务必按 PLAN.md §4.3 的 5 场景分布采样起点（A 短距25% / B 中距25% / C 远距探索25% / D 多地标干扰15% / E 起点即终点10%），并随机化起点方位角和起点 Yaw，避免模型过拟合固定模式。

### 2.1 批量驱动：高度采样 + 场景动态兜底 (PLAN §4.1.1 + §4.3.1)

单条 `generate_trajectory()` 之外，批量层必须实现两套采样器：(1) 高度档位采样，(2) 场景比例动态兜底。两者**互不依赖**，串联调用即可。

**关键提醒**：

- `compute_scenario_ratios` 必须在批量开始前**只跑一次**（扫描全图 has_landmark_class 较慢），缓存结果。
- 高度档位采样独立于场景采样——高度按视觉尺度分桶（不挑图），场景按图像角色分桶（挑图），两者交叉笛卡尔。
- 实际生成完后，若某场景偏差 >3% 立刻调整：通常原因是 `random.choice(l_class_pids)` 在 C/D 场景下被高频回采（少数 L 类图被反复用），需要把回采限制为"每张图最多 N 条 C/D 轨迹"再补充。
- **instruction snapshot 语义（防 OOD 关键）**：训练 `batch_generate` 与评估 `run_evaluation_episode` **必须**在各自循环外只调用一次 `compute_instruction_context_at_start` + `build_instruction`，整条轨迹共享同一份指令文本。任何"每步重算 spatial_word / target_count_in_view" 的实现都是 train-eval 分布漂移（PLAN §4.2.2 line 217: "spatial_word 由调用方根据**起点视野下** target 相对当前视野中心的方位预先计算"）。
- `sample_episode` 在 `max_attempts` 内若采不到合法 target/landmark 组合即返回 None，调用方再换 pid 重试 `max_pid_retries` 次。避免无限循环，同时让稀疏场景（C/D 在 small-vehicle 主导的数据上）能稳定退化到下一轮。


## 3. 单阶段 SFT 训练流程与命令

建议使用 `LLaMA-Factory`。**采用单阶段全量 SFT**：所有字段（2 张图 + 5 步历史动作 + 累计位移 + CoT + Action）一次性训练，不再做 "Stage 1 mask 字段 + Stage 2 放开" 的两阶段课程（理由见 PLAN §2.1 / §5.2 的反向 curriculum 风险）。

### 3.1 数据集划分（按 DOTA 图 ID，防泄露）

`train.jsonl` 与 `eval.jsonl` 必须**按 DOTA 图像 ID 划分**（90% : 10%，整张图作为最小划分单位），不能用 LLaMA-Factory 的 `val_size` 随机切——同一张 DOTA 大图会衍生出空间相邻的多条轨迹，随机切会让验证集和训练集共享同张大图，闭环 SR 评估失真。详见 PLAN §5.1。
在 `data/dataset_info.json` 中同步注册两个数据集

### 3.2 单阶段 SFT 配置

`qwen2_vl_sft.yaml`

运行命令：
```bash
llamafactory-cli train qwen2_vl_sft.yaml
```

### 3.3 可选：数据顺序课程（不改字段、仅改采样比例）

若担心远距探索样本收敛慢，可在单阶段内按 epoch 调整起点场景采样比例（PLAN §5.2）：

| Epoch | A 短距 | B 中距 | C 远距 | D 干扰 | E 即终点 | 目的 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 50% | 10% | 10% | 5% | 25% | 先学会 STOP 与精细对准 |
| 2-3 | 25% | 25% | 25% | 15% | 10% | 恢复 PLAN §4.3 标准比例 |

这是**数据顺序课程**——prompt 骨架始终不变，模型不会学到"忽略字段"的错误捷径，与早期废弃的字段 mask 课程本质不同。

## 4. 闭环评估脚本 (`evaluate.py`)

微调完成后，我们需要编写一个仿真闭环代码，验证模型是否能真的“飞”到目标。

## 5. 基准测试与指标计算 (Metrics & Baselines)

为了对比你的微调模型表现，需要评估 Random、Rule-based (Greedy) 和 Zero-shot 基准，并计算核心指标 (Success Rate, SPL, Avg Steps)。

## 6. 脚本编写流程与注意事项 (Scripting Workflow & Precautions)

为了确保数据生成和闭环测试顺利进行，编写这套 Python 脚本时需要遵循以下流程和注意事项：

### 6.1 推荐的开发流程
1.  **单图跑通跑通 (Sanity Check)**：
    *   先不要批量处理 DOTA 的所有标注文件。挑一张特征明显的图片（如 `P0001.png`），手动硬编码一个起点的 $(X, Y)$ 和目标坐标。
    *   写好 `generate_trajectory` 里的数学坐标映射和 OpenCV 旋转裁剪逻辑。
    *   在每一 `step` 将切下来的图用 `cv2.imshow` 或保存到本地查看，**肉眼确认**：裁剪出来的图是否真的把目标保持在“前方/右方/左方”对应的视野内；当 `yaw` 发生改变时，正前方是否在图片的最上方。这是最核心、也是最容易出错的一步。
2.  **批量脚本封装与多进程加速**：
    *   确认单条轨迹生成无误后，编写读取全量 `DOTA/train` 标注文件的主循环。
    *   由于图像的旋转和裁剪纯靠 CPU 计算，强烈建议使用 Python 的 `multiprocessing.Pool` 来并行生成数据，将几十个小时的数据生成时间压缩到 1-2 小时。
3.  **日志与异常处理 (Try-Except)**：
    *   DOTA 数据集有极个别标注可能在图像边缘。如果是边缘目标，外接圆裁剪 (`extend_size`) 可能会抛出数组越界错误 (`IndexError`)。
    *   一定要在切图函数外包裹 `try-except`，遇到边缘切图失败直接 `continue` 放弃这条轨迹，千万不要让脚本因为个别脏数据中途崩溃。

### 6.2 极易踩坑的注意事项 (Pitfalls)
*   **黑边问题与矩阵旋转机制**：如果你先裁下 $224 \times 224$ 再进行 $45^\circ$ 旋转，不仅会丢掉角落视野，还会出现黑色三角区。**一定要先裁下 $224 \times \sqrt{2} \approx 317$ 的大图，对大图进行旋转，然后再从正中心裁出 $224 \times 224$ 的小图。**
*   **局部坐标系与偏航角 (Yaw) 体系的对应关系**：
    *   本方案中规定 `Yaw=0` 是正北，顺时针为正。
    *   在通过 $\vec{v}$ (X向量, Y向量) 计算 `target_yaw` 时，要时刻警惕 OpenCV 或图像坐标系与传统笛卡尔坐标系的差异（OpenCV 中 Y 轴是向下的）。为了省去脑筋急转弯，最简单的方式是**统一把图像的中心点当作笛卡尔坐标原点 (0,0)，Y 轴向上为正，X 轴向右为正**，所有的几何计算都在笛卡尔坐标系完成，最后切图时再映射回像素行列系。
*   **数据泄露 (Data Leakage) 预防（与 PLAN §5.1 完全对齐）**：
    *   **划分单位**：数据划分发生在 **DOTA Train 文件夹 (`data/train/`) 内部**——按 P-id 90:10 切为 `image_ids_train.txt` / `image_ids_val.txt`，整张图作为最小划分单位。`train.jsonl` 与 `eval.jsonl` 据此分别生成，**两者绝不共享任何 P-id**。
    *   **DOTA 官方 val (`data/val/`) 的定位**：作为**额外的 out-of-distribution 测试集**单独评估，不参与上述 90:10 划分。报告时分两栏：(a) `image_ids_val` 内部留出测试（in-distribution）；(b) DOTA 官方 val（OOD）。这样既验证模型在训练集内部留出图像上的泛化，又验证完全独立分布上的泛化。
    *   **早期文档的矛盾说法**："`evaluate.py` 必须并且只能在 Val 文件夹里挑选图片"已废弃——闭环 SR 评估的主战场是 `image_ids_val`（与训练验证 loss 共用同一组图），DOTA 官方 val 只作为补充的 OOD 报告。

### 6.3 无人机 VLN 视觉语言导航系统架构与开发流程

基于 DOTA 数据集与 Qwen2.5-VL-7B，采用**单阶段 SFT 微调**策略实现全流程从数据合成、模型微调到闭环评测（可选叠加数据顺序课程，但 prompt 骨架始终一致；废弃两阶段字段 mask 课程的理由见 PLAN §2.1）。

**Steps**

**Phase 1: 数据生成引擎 (Data Pipeline)**
0. **数据可用性核查（必做的前置步骤）**：调用 `build_pid_to_path_index(data_root, split)` 扫描 `data/train/` 与 `data/val/`，构建 `{P-id: (img, label, meta)}` 三件套索引并打印 stats。**仅在三件套齐全的 P-id 上跑后续流程**——本地数据已知有 ~77% 标注无对应图像，必须 inner-join 后再均摊轨迹规模，避免 `cv2.imread` 静默返回 None。
1. 实现 DOTA 标注解析与坐标映射逻辑（**训练端 GSD 抖动 `GSD ∈ [0.35, 0.7]` + 评估端固定 `GSD=0.5` (ppm=2.0)**，不再做按图动态裁剪或 GSD 过滤，理由见 PLAN §4.1：本地 98.8% train 图 / 94.8% val 图 meta 标 `gsd:None`，"按图动态裁剪"99% 走 default 兜底，等价于一刀切；meta 仍读取但仅作 trajectory 元信息记录。**调用契约**：`pixel_per_meter()` 仅在轨迹/episode 入口调一次，显式传给所有下游函数，严禁内部再次调用）。
2. 实现空间词语生成器 (`get_spatial_word`) 以及基于"探索-逼近"规则的思维链 (CoT) 生成。
3. 编写无人机飞行轨迹仿真模块，涵盖：先旋转后移动的动力学模型（**基于模型 action 累加相对偏航 `atan2(grid_x, grid_y)`，不直接对准 target_yaw**）、视场角映射截图、图像无黑边旋转剪裁以及高斯物理噪声注入。
4. 全量输出 ShareGPT 多模态格式的数据集。

**Phase 2: 数据集划分与注册（防泄露，关键）**
1. *depends on 1* 按 DOTA 图像 ID 90% : 10% 划分为 `image_ids_train.txt` / `image_ids_val.txt`，整张图作为最小划分单位（PLAN §5.1）。
2. 分别用两个图 ID 列表生成 `train.jsonl` 与 `eval.jsonl`，**两者绝不共享任何图像**；在 `data/dataset_info.json` 中注册 `uav_full` 与 `uav_eval`。

**Phase 3: 单阶段 SFT 训练管道 (基于 LLaMA-Factory)**
1. *depends on 2* 编写 `qwen2_vl_sft.yaml`，设定 LoRA 参数（rank=64 / alpha=128，仅训 LLM 端 q/k/v/o/gate/up/down_proj，`freeze_vision_tower: true`）。
2. 一次性训练 2-3 个 epoch，所有字段（2 张图 + 5 步历史动作 + 累计位移 + CoT + Action）全量参与；`eval_dataset: uav_eval` 显式指定按图划分的独立验证集，**禁止用 `val_size` 随机切**。
3. 可选：通过 epoch 内的起点场景采样比例做**数据顺序课程**（非字段 mask 课程，避免反向 curriculum 风险，详见 §3.3）。

**Phase 4: 闭环评测系统 (Evaluation Engine)**
1. *depends on 3* 在 `eval.py` 中实现自回归推理循环，模拟未见环境下的真实飞行步骤。**双轨评估（PLAN §5.1 修订）**：
   - **训练 loss 监控集**: `image_ids_val.txt` 对应的 `data/train/` 留出图像（42 张，整张图划分，仅用于训练期 loss 曲线观察防过拟合，**不作 SR 主指标依据**）
   - **闭环 SR 主指标集**: `data/val/` 官方 val（135 张全部三件套齐全，作为最终 SR/SPL 报告主战场）
2. 计算成功率 (SR)、路径加权成功率 (SPL) 以及平均成功步数；按多高度档位分别给 SR@25m / SR@50m / SR@75m。模型不知道当前高度，评估器按生成 episode 时的 `height_m` 元信息分桶报指标。

**Relevant files**
- train_data_generate.py — 实现数据全量解析、CoT 逻辑构建、物理引擎与 JSONL 导出，按 DOTA 图 ID 划分输出 `train.jsonl` + `eval.jsonl`。
- airsim/AirNav/training/qwen2_vl_sft.yaml (需创建) — 单阶段 SFT 配置：LoRA rank=64、冻结 ViT、`dataset: uav_full` + `eval_dataset: uav_eval`（按 DOTA 图划分的独立验证集）。
- eval.py — 进行大模型流式调用与位姿环境交互计算。

**Verification**
1. 测试轨迹生成：在截取的 10 张小规模 DOTA 图像上运行 `train_data_generate.py`，肉眼核对生成的 action、thought 与 偏航角视角剪裁是否合理；确认 yaw 累加方向与代码 `atan2(grid_x, grid_y)` 一致。
2. 校验文件格式：检验输出 JSONL 的图片路径合法性以及特殊标记 `<image>` 的占位计数（== `images` 数组长度）；确认 `train.jsonl` 与 `eval.jsonl` 的图 ID 集合无交集。
3. LLaMA-Factory 冒烟测试：使用 `max_samples: 100` 小样本集跑一次单阶段 SFT，确保能够正常加载多模态特征并计算 Loss。
4. 初始基线测试：挂载原始的无微调 Base 模型，调用 `eval.py` 生成 Random & Zero-shot 基础对照 SPL，再代入最终调优模型。

**Decisions**
- **训练范式：单阶段全量 SFT**：所有字段（2 张图 + 5 步历史动作 + 累计位移 + CoT + Action）一次性参与训练，**不再做** "Stage 1 mask 字段 + Stage 2 放开" 的两阶段课程——该设计会教会模型"忽略历史 / 累计位移"字段（反向 curriculum），与最终目标直接冲突（详见 PLAN §2.1 / §5.2）。若需 curriculum，仅通过单阶段内的**起点场景采样比例**实现，prompt 骨架始终不变。
- **防 OOM 设定**：历史图像在输入构造中严格保持上限 1 张 $(T-1)$ 拼接当前帧 $(T)$。
- **Action 输出约束**：坚决摒弃绝对地理坐标系预测，保持在视野尺度内的 $\pm 5$ 相对有限网格输出，增强泛化性能。模型只输出局部网格 (x, y)，飞控脚本解析为旋转角度 + 移动距离（无需 Z / Yaw 输出）。
- **训练降本与 sim-to-real 平衡（PLAN §5.2 修订）**：单阶段 SFT 采用 **`freeze_vision_tower: true` + `freeze_multi_modal_projector: false`** 组合：ViT (~600M) 冻结防 OOM 与防劣化，projector (~30M) 全参解冻以缩小 DOTA 卫星图 → 无人机航拍 domain gap。projector 是 ViT→LLM 的语义对齐瓶颈层，解冻它显存增量 <5% 但跨域收益显著，是除 §9 真机数据微调外最便宜的域适配手段。LLaMA-Factory YAML 仅有 `freeze_vision_tower` / `freeze_multi_modal_projector` / `freeze_language_model` 三个整体冻结开关，**没有原生按层冻结 ViT 的参数**；若坚持要训 ViT 后段，需自写训练 callback 手动改 `requires_grad`，非 YAML 配置项。**真机数据微调阶段** (§9) projector 改回冻结（少量数据易过拟合并破坏跨域对齐）。
- **多高度数据生成**：当前训练数据按 25m / 50m / 75m = 50% / 30% / 20% 比例采样，对应 `grid_size_m = 5 / 10 / 15`、`fov_m = 50 / 100 / 150`；起点距离按高度等比例放大。**prompt 不暴露飞行高度**，多高度信号仅隐式编码在像素尺度中。
- **场景采样动态兜底（PLAN §4.3.1）**：本地 small-vehicle 占 63%，约 60-70% 图无 L 类地标，导致场景 C/D 名义占比 40% 超过实际可达 ~33%。批量生成必须先扫描 `has_landmark_class(pid)`，按 `compute_scenario_ratios()` 把缺额匀给场景 B（不依赖 L 类），并在生成后输出实际占比统计表，偏差 >3% 调整下一轮。
- **Turn-then-Move 物理模型**：训练数据 yaw 更新必须用模型 action 推出的相对偏航 `atan2(grid_x, grid_y)` 累加到 current_yaw，**禁止直接对准 active_target 的全局 yaw**——后者会让 (±5 截断) 后实际飞行方向 ≠ action 描述方向，且训练标签偏向 (0, +N) 失去转弯能力。该约束在 `generate_trajectory()` 与 `update_physics_and_draw()` 中必须完全一致（详见 PLAN §4.3）。
- **STOP 阈值（训练-评估完全一致）**：训练标签 STOP 半径 = 评估 SR 阈值 = `max(5m, 0.5 × grid_size)`（详见 PLAN §1.3 / §3.2）。代码统一通过 `stop_eps_m(grid_size_m)` 函数获取，避免散落硬编码。
- **(0,0) 与 STOP 等价**：训练数据中 80% 输出 `Action: STOP`、20% 输出 `Action: (0, 0)` 混合采样，飞控/评估脚本解析层完全等价处理。
- **数据划分防泄露**：按 DOTA 图像 ID 90:10 划分 `train.jsonl` / `eval.jsonl`，整张图作为最小划分单位；YAML 用 `eval_dataset: uav_eval` 显式指定独立验证集，**禁止用 `val_size` 随机切**（详见 PLAN §5.1）。
- **累计位移：yaw 积分而非局部代数和**：每步保存 yaw + action 序列，按全局北/东向积分（详见 PLAN §1.2）。早期"前进/右移代数和"方案在 yaw 变化时不具备物理意义，已废弃。
- **历史动作 5 步 + 累计状态向量**：仅基于模型自身历史输出累加（非 GPS 真值），保证仿真和实机部署可获取的特征完全一致。

## 7. 项目展示建议
*   **亮点展示**：重点强调使用了 DOTA 降维模拟 3D 飞行，极大降低了数据获取成本。强调使用了**数据顺序课程（场景比例采样）+ 物理噪声注入 + 累计位移 yaw 积分**等数据增强与表征设计（注意：避免宣传"两阶段字段 mask 课程"——该方案已废弃，详见 §6.3 Decisions）。
*   **可视化**：在闭环评估 `evaluate.py` 运行时，使用 OpenCV 把无人机的真实轨迹画在 DOTA 大图上。最终生成一个 GIF：一个红点（无人机）随着模型的逐步推理，在卫星地图上不断转向、移动，最终准确悬停在目标（如网球场）正上方。

## 8. 部署方案：远端推理 + 机载飞控

Qwen2.5-VL-7B FP16 显存约 14GB，绝大部分消费级 / 工业级机载计算单元（Jetson Orin Nano 8GB、Pixhawk 等）无法本地推理。本项目采用 **"远端推理 + 5G/WiFi 回传，机载只做飞控"** 方案。这与"悬停拍摄"的飞行模式天然契合——推理期间无人机悬停，不存在"图传到地面后位置已经偏移"的问题。

### 8.1 系统拓扑

```
[无人机机载]                                [地面 GPU 工作站]
摄像头 → JPEG 压缩 →
        ↓ WiFi 5GHz / 5G CPE / 4G
        ─────────────────→  vLLM 部署的 Qwen2.5-VL-7B
        ←─────────────────  Action: (gx, gy) / STOP
脚本解析:
  local_delta_yaw = atan2(gx, gy)
  dist_m = sqrt(gx²+gy²) × grid_size
↓
PX4 / ArduPilot / DJI SDK (MAVLink):
  1. 旋转 local_delta_yaw 度
  2. 沿新航向前进 dist_m 米
  3. 悬停, 触发下一帧拍摄
```

### 8.2 关键链路设计

| 项 | 推荐配置 | 说明 |
| --- | --- | --- |
| 通信 | WiFi 5GHz / 5G CPE | 视距 200-500m, 延迟 ~10ms |
| 图像编码 | JPEG quality=80, ~50KB/张 | 2.4Mbps 带宽即可 |
| 推理服务 | vLLM + AWQ INT8 量化 | 4090 单卡 5-10 req/s, 精度损失 <2% |
| 决策频率 | 0.2-0.5 Hz | 悬停-拍摄-推理-移动循环节拍 |
| 超时策略 | 单步 >5s 触发悬停重试; 连续 3 次失败返航 | 失联保护 |

### 8.3 飞控侧动作解析（脚本伪代码）

```python
def execute_vlm_action(gx: int, gy: int, grid_size_m: float, drone):
    """
    模型输出局部网格 (右, 前) -> 旋转 + 直线飞行。
    """
    if (gx, gy) == (0, 0):           # STOP 同义
        drone.hover()
        return

    # 1. 计算相对当前朝向的旋转角 (顺时针为正)
    delta_yaw_deg = math.degrees(math.atan2(gx, gy))
    # 2. 移动距离
    dist_m = math.hypot(gx, gy) * grid_size_m

    # 3. Turn-then-Move (悬停期间执行)
    drone.set_yaw_relative(delta_yaw_deg)
    drone.wait_until_yaw_settled(tolerance=2.0)   # 等待 yaw 稳定
    drone.move_forward(dist_m)
    drone.wait_until_position_settled()           # 等待平移到位 + GPS 回正
    drone.hover()
    time.sleep(0.5)                               # 悬停稳定后再拍下一帧
```

### 8.4 安全护栏（必须实装）

*   **Geofence（电子围栏）**：飞控侧硬限制，超出预设区域强制悬停
*   **动作合法性二次校验**：丢弃 VLM 异常输出（如 (99, -99) 或与累计位移严重矛盾的动作）
*   **低电量返航**：电量 <30% 自动返航
*   **失联保护**：失联 >10s 悬停，>30s 降落
*   **STOP 兜底**：连续 5 步动作变化 <1 网格则强制 STOP，避免在目标附近震荡

### 8.5 降本备选

| 方案 | 模型 | 显存 | 单步延迟 | 场景 |
| --- | --- | --- | --- | --- |
| A. 远端 FP16（推荐）| Qwen2.5-VL-7B | 14GB | 1-3s | 实验室 / 大创演示 |
| B. 远端 INT8 | Qwen2.5-VL-7B + AWQ | 8GB | 0.6-1.5s | 高吞吐多机协同 |
| C. 蒸馏小模型 | Qwen2-VL-2B | 4GB | 0.3-0.8s | 远端算力较弱 |
| D. 机载部署 | Qwen2-VL-2B + INT4 | 2GB | 0.5-1.5s | Jetson Orin NX 16GB |

---

## 9. 真实数据微调（Sim-to-Real 最后一公里）— **Stretch Goal / 视项目时间余量决定**

> **定位说明**：本节为**可选 Stretch Goal**，不属于大创核心交付。是否实施取决于项目时间余量、实机硬件可用性和指导老师的优先级判断。**核心交付（DOTA 仿真数据 + 单阶段 SFT 微调 + 闭环 SR/SPL 评估）已足以构成完整大创成果**——本节是"锦上添花"的真机演示部分。详见 PLAN §7 的完整风险评估。
>
> **跳过时的兜底**：用 §4 的闭环 evaluate.py 在 DOTA 验证集上生成飞行 GIF，作为最终演示素材即可。

**如果决定实施**：仅用 DOTA 仿真训练的模型，实机直接零样本部署 SR 通常仅 5%-20%（域差距严重）。补充少量真实数据微调可将实机 SR 提升到 50%-70%。

### 9.1 数据采集（轻量）

*   **采集设备**：最终部署用的同一台无人机（保证相机参数一致）
*   **场景数量**：3-5 个不同场景（学校操场、停车场、小区花园），每场景 50-100 条轨迹
*   **总规模**：200-500 条轨迹即可，每条 5-10 步，总样本量约 2000-5000 个 (image, action) 对
*   **标注简化**：
    *   不需要精细 CoT，只需 `(图, 任务文本, 正确 action)` 三元组
    *   **半自动标注流程**：
        1. 手动遥控无人机飞到目标，记录每一步无人机自身位移 $(dx_m, dy_m)$ 和 yaw 变化
        2. 按 `grid_size_m` 离散化为 $(g_x, g_y)$ 作为 action 标签
        3. 可选：用单阶段 SFT 模型先自动生成 CoT，人工只校验和修正错误样本（active learning）

### 9.2 真实数据微调脚本

`qwen2_vl_real_finetune.yaml`:
```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-7B
adapter_name_or_path: saves/qwen2_vl-7b/lora/uav_sft  # 加载单阶段 SFT 完整 LoRA
image_max_pixels: 200704
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 32                      # rank 减半, 避免少量数据过拟合
lora_alpha: 64
lora_dropout: 0.1
freeze_vision_tower: true          # 真实数据少, 不再动 ViT, 防遗忘
freeze_multi_modal_projector: true # 真实数据微调阶段 projector 也冻结
                                   # 理由: §5.2 阶段已用大规模仿真数据训过 projector,
                                   #       此处真实数据仅 200-500 条, 解冻 projector 易过拟合并破坏跨域对齐。
                                   #       与 §5.2 阶段的 false 相反 (那时是大数据 + 强需要域适配)。
freeze_language_model: false       # LLM 走 LoRA 微调

### dataset
dataset: uav_real_mix              # 真实:仿真 = 1:4 混合, 防止灾难性遗忘
template: qwen2_vl
cutoff_len: 4096

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-5              # 远低于单阶段 SFT, 仅做微调防遗忘
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
weight_decay: 0.01
bf16: true
gradient_checkpointing: true
```

### 9.3 验证流程

1.  **留出测试集 (held-out)**：从真实数据中划出 20%-30% 作为评估集，**不参与训练**
2.  **AB 对比**：分别评估"仅单阶段 SFT 模型"vs"单阶段 SFT + 真实微调"在留出集上的 SR / SPL，差值即为 sim-to-real 微调的增益（这是大创成果的关键证据）
3.  **真实飞行素材**：在 3-5 个新场景下做实机飞行，录视频 + 叠加轨迹图，作为最终展示
