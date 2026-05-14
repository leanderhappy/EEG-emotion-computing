# DEAP 任务三：跨被试迁移学习

## 简介

本目录实现基于多种模型的跨被试情感分类迁移实验，支持 MLP (DANN)、SVM 和 Random Forest。实验采用 `source_subject -> target_subject` 的 pairwise 设置：对每个 target subject，使用无标签 EEG 特征选择分布最相似的 source subject，并评估无迁移和迁移方法的效果。

### MLP (DANN) 方法

```text
source_subject A 用于训练
target_subject B 用于测试
no_transfer：使用 DANNModel 的特征提取器和情绪分类器，只用 A 的数据和情感标签训练
with_transfer：使用同一个 DANNModel，额外使用 A+B 的域标签做 DANN 迁移，情感标签只用 A
```

### SVM/RF 方法 (ASFM)

SVM 和 RF 使用 ASFM (Adaptive Subspace Feature Mapping) 迁移方法：

```text
1. DI 特征选择：结合互信息 (MI) 和 KS 检验，选择域不变特征
2. SVD 旋转对齐：通过 SVD 分解对齐源域和目标域子空间
3. 重要性加权：估计源域样本重要性权重
4. Score Blending (SVM)：融合对齐模型和原始模型的预测
5. Trial 聚合：将窗口级预测聚合为 trial 级预测
```

## 目录结构

```text
Task3-transfer_learning/
├── src/
│   ├── dann_model.py            # DANN / GRL / MLP 模型定义
│   ├── mlp_dann_transfer.py     # MLP 与 DANN-MLP pairwise 实验入口
│   ├── run_svm_transfer.py      # SVM + ASFM 迁移实验
│   ├── run_rf_transfer.py       # Random Forest + ASFM 迁移实验
│   └── mlp_plot_results.py      # 迁移实验结果可视化脚本
├── output/
│   ├── figures/                 # 可视化结果图与 summary_statistics.csv
│   └── <data_source>/
│       └── <task>/
│           ├── mlp/
│           │   ├── no_transfer/
│           │   └── with_transfer/
│           ├── svm-asfm/
│           │   └── <split_variant>/
│           │       └── <pair_tag>/
│           └── rf-asfm/
│               └── <split_variant>/
│                   └── <pair_tag>/
└── README.md
```

## 实验设置

每次实验为每个 target subject 选择一个 source subject：

```text
source_subject -> target_subject
```

选择规则：

```text
source_subject != target_subject
source_subject 的情绪标签至少包含两个类别
使用 source/target 的无标签 EEG 特征计算 CORAL distance
选择 CORAL distance 最小的 source_subject
```

CORAL distance 比较两个被试特征协方差的差异，不使用 target 标签，因此符合无监督跨被试迁移设置。

两种方法使用同一批 pair：

| 方法 | 训练使用 | 测试使用 |
|------|----------|----------|
| `no_transfer` | source A 的 EEG + 情感标签 | target B |
| `with_transfer` | source A 的情感标签，A/B 的 EEG 与域标签 | target B |


## 运行命令

### MLP (DANN)

默认同时运行无迁移 MLP 和 DANN-MLP：

```powershell
python Task3-transfer_learning/src/mlp_dann_transfer.py --data_root task1 --task binary
```

只运行无迁移 MLP：

```powershell
python Task3-transfer_learning/src/mlp_dann_transfer.py --data_root task1 --task binary --only no_transfer
```

只运行 DANN-MLP：

```powershell
python Task3-transfer_learning/src/mlp_dann_transfer.py --data_root task1 --task binary --only dann
```

自定义训练参数：

```powershell
python Task3-transfer_learning/src/mlp_dann_transfer.py --data_root task1 --task binary --epochs 150 --batch_size 16 --lr 5e-4 --lambda_max 0.3 --pretrain_epochs 20
```

### SVM (ASFM)

运行 SVM + ASFM 迁移实验：

```powershell
python Task3-transfer_learning/src/run_svm_transfer.py --data_root task1 --task binary
```

指定特定 source/target subject：

```powershell
python Task3-transfer_learning/src/run_svm_transfer.py --data_root task1 --task binary --source_subject s01 --target_subject s02
```

启用 GridSearch 自动调参：

```powershell
python Task3-transfer_learning/src/run_svm_transfer.py --data_root task1 --task binary --use_gridsearch
```

### Random Forest (ASFM)

运行 RF + ASFM 迁移实验：

```powershell
python Task3-transfer_learning/src/run_rf_transfer.py --data_root task1 --task binary
```

自定义 RF 参数：

```powershell
python Task3-transfer_learning/src/run_rf_transfer.py --data_root task1 --task binary --rf_max_depth 15 --rf_min_samples_split 3
```

### 绘制结果

```powershell
python Task3-transfer_learning/src/mlp_plot_results.py
```

## 参数说明

### MLP / DANN 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `task1` | `task1` / `official` / 自定义路径 |
| `--output_root` | `Task3-transfer_learning/output` | 输出根目录 |
| `--task` | `binary` | `binary` / `threeclass` |
| `--target` | `valence` | `valence` / `arousal` |
| `--seed` | `42` | 模型训练随机种子 |
| `--only` | `both` | `both` / `no_transfer` / `dann` |
| `--epochs` | `120` | Torch MLP / DANN 训练轮数 |
| `--batch_size` | `16` | Torch MLP / DANN batch size |
| `--lr` | `5e-4` | Torch MLP / DANN 学习率 |
| `--lambda_max` | `0.3` | GRL 对抗强度上限 |
| `--hidden` | `32` | Torch MLP / DANN 隐藏层维度 |
| `--latent` | `32` | Torch MLP / DANN latent feature 维度 |
| `--pretrain_epochs` | `20` | 只用 source 情绪标签预训练的轮数 |
| `--weight_decay` | `1e-4` | Adam weight decay 正则强度 |
| `--grad_clip` | `5.0` | Torch MLP / DANN 梯度裁剪阈值，设为 `0` 可关闭 |
| `--log_interval` | `20` | DANN 训练日志间隔，设为 `0` 可关闭 |

### SVM / RF 公共参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | `task1` | `task1` / `official` / 自定义路径 |
| `--output_root` | `Task3-transfer_learning/output` | 输出根目录 |
| `--task` | `binary` | `binary` / `threeclass` |
| `--seed` | `42` | 模型训练随机种子 |
| `--feature_k` | `18` (SVM) / `22` (RF) | DI 特征选择数量 |
| `--asfm_d` | `22` | ASFM 子空间维度 |
| `--asfm_window_seconds` | `4.0` | 窗口长度 (秒) |
| `--asfm_window_step_seconds` | `2.0` | 窗口步长 (秒) |
| `--asfm_source_weight_clip_min` | `0.5` | 源域重要性权重下限 |
| `--asfm_source_weight_clip_max` | `2.0` | 源域重要性权重上限 |
| `--num_repeats` | `1` | 每对 pair 重复实验次数 |
| `--num_random_targets` | `32` (SVM) / `0` (RF) | 随机目标数量，`0` = LOTO (所有被试) |
| `--source_select_max_samples` | `400` | 源域选择最大采样数 |
| `--source_select_coarse_topk` | `5` | 粗筛候选源数量 |
| `--source_subject` | `None` | 手动指定源被试 |
| `--target_subject` | `None` | 手动指定目标被试 |
| `--use_gridsearch` | `False` | 启用 GridSearchCV 自动调参 |
| `--di_alpha` | `0.5` | DI 特征选择 alpha：MI 权重 |

### SVM 特有参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--asfm_num_pl_iterations` | `3` | 伪标签迭代次数 |
| `--asfm_pseudo_tau` | `0.75` | 伪标签置信度阈值 |
| `--asfm_pseudo_tau_decay` | `0.05` | 每次迭代 tau 衰减量 |
| `--asfm_pseudo_tau_min` | `0.55` | tau 最小值 |
| `--asfm_pseudo_weight` | `0.5` | 伪标签样本权重 |
| `--asfm_max_pseudo_per_class` | `30` | 每类最大伪标签数 |
| `--asfm_pseudo_imbalance_ratio` | `2.0` | 伪标签类别不平衡比率 |
| `--asfm_pseudo_single_class_cap` | `8` | 单类别时伪标签上限 |

### RF 特有参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--rf_max_depth` | `10` | RF 最大树深度 |
| `--rf_min_samples_split` | `5` | RF 最小分裂样本数 |

## 输出结果

### MLP 输出目录

无迁移 MLP：

```text
Task3-transfer_learning/output/<data_source>/<task>/mlp/no_transfer/
```

DANN-MLP：

```text
Task3-transfer_learning/output/<data_source>/<task>/mlp/with_transfer/
```

### SVM 输出目录

```text
Task3-transfer_learning/output/<data_source>/<task>/svm-asfm/<split_variant>/<pair_tag>/
```

### RF 输出目录

```text
Task3-transfer_learning/output/<data_source>/<task>/rf-asfm/<split_variant>/<pair_tag>/
```

### 可视化结果

```text
Task3-transfer_learning/output/figures/
```

MLP 目录会生成：

| 文件 | 内容 |
| --- | --- |
| `pairwise_results.txt` | 每组 pair 准确率、总体均值/方差、classification report |
| `pairwise_accuracy.csv` | `pair_id,source_subject,target_subject,selection_distance,accuracy` |
| `pairwise_confusion_matrix.png` | 汇总混淆矩阵 |

SVM/RF 目录会生成：

| 文件 | 内容 |
| --- | --- |
| `results.txt` | 实验配置、每组 pair 准确率、总体均值/方差、classification report |
| `repeat_results.csv` | 每次重复的详细结果 |
| `pair_summary.csv` | 每对 pair 的汇总统计 |
| `source_selection_summary.csv` | 源域选择过程记录 |
| `source_only_confusion_matrix.png` | 无迁移混淆矩阵 |
| `transfer_confusion_matrix.png` | 迁移后混淆矩阵 |

`figures` 目录会生成：

| 文件 | 内容 |
| --- | --- |
| `*_pair_line.png` | 每个 target subject 上无迁移与 DANN 的准确率折线对比 |
| `*_delta_bar.png` | 每个 target subject 的迁移提升量 |
| `*_distance_delta_scatter.png` | CORAL distance 与迁移提升之间的关系 |
| `summary_mean_accuracy.png` | 不同数据源和任务上的总体平均准确率对比 |
| `summary_statistics.csv` | 各实验设置的均值、方差、提升/下降 pair 数统计 |
