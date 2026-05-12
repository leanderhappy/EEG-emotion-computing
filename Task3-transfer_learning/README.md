# DEAP 任务三：跨被试迁移学习

## 简介

本目录实现基于 MLP 的跨被试情感分类迁移实验。实验采用 `source_subject -> target_subject` 的 pairwise 设置：对每个 target subject，使用无标签 EEG 特征选择分布最相似的 source subject，并用同一批 pair 同时评估同架构无迁移 MLP 和 DANN-MLP。

基本流程：

```text
source_subject A 用于训练
target_subject B 用于测试
no_transfer：使用 DANNModel 的特征提取器和情绪分类器，只用 A 的数据和情感标签训练
with_transfer：使用同一个 DANNModel，额外使用 A+B 的域标签做 DANN 迁移，情感标签只用 A
```

## 目录结构

```text
Task3-transfer_learning/
├── src/
│   ├── dann_model.py            # DANN / GRL / MLP 模型定义
│   ├── mlp_dann_transfer.py     # MLP 与 DANN-MLP pairwise 实验入口
│   └── mlp_plot_results.py          # 迁移实验结果可视化脚本
├── output/
│   ├── figures/                 # 可视化结果图与 summary_statistics.csv
│   └── <data_source>/
│       └── <task>/
│           └── mlp/
│               ├── no_transfer/
│               └── with_transfer/
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

绘制实验结果图：

```powershell
python Task3-transfer_learning/src/mlp_plot_results.py
```

## 参数说明

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

## 输出结果

无迁移 MLP：

```text
Task3-transfer_learning/output/<data_source>/<task>/mlp/no_transfer/
```

DANN-MLP：

```text
Task3-transfer_learning/output/<data_source>/<task>/mlp/with_transfer/
```

可视化结果：

```text
Task3-transfer_learning/output/figures/
```

每个目录会生成：

| 文件 | 内容 |
|------|------|
| `pairwise_results.txt` | 每组 pair 准确率、总体均值/方差、classification report |
| `pairwise_accuracy.csv` | `pair_id,source_subject,target_subject,selection_distance,accuracy` |
| `pairwise_confusion_matrix.png` | 汇总混淆矩阵 |

`figures` 目录会生成：

| 文件 | 内容 |
|------|------|
| `*_pair_line.png` | 每个 target subject 上无迁移与 DANN 的准确率折线对比 |
| `*_delta_bar.png` | 每个 target subject 的迁移提升量 |
| `*_distance_delta_scatter.png` | CORAL distance 与迁移提升之间的关系 |
| `summary_mean_accuracy.png` | 不同数据源和任务上的总体平均准确率对比 |
| `summary_statistics.csv` | 各实验设置的均值、方差、提升/下降 pair 数统计 |
