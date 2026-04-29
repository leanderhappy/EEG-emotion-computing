# DEAP 任务一预处理与任务二数据导出

本目录是课程任务一的独立工程：从 DEAP 原始 `BDF` 文件开始做预处理，并把清洗后的结果整理成任务二建模可直接读取的数据。

## 目录结构

```text
Task1-preprocess/
├── data/
│   ├── raw/DEAP/data_original/      # 放 DEAP 原始 BDF：s01.bdf ... s32.bdf
│   ├── raw/DEAP/metadata_csv.zip    # 放 DEAP 元数据，任务二标签导出需要
│   ├── interim/                     # 任务一中间表：通道表、trial 事件、伪迹日志
│   ├── processed/                   # 任务一清洗结果：epochs_clean.fif、trial_qc.csv
│   └── task2/                       # 任务二导出结果：标签、npz、索引表
├── output/                          # 任务一运行汇总和报告用图
├── src/
│   ├── run_task1.py                 # 执行任务一预处理
│   ├── prepare_task2_labels.py      # 只导出任务二 trial 标签
│   ├── prepare_task2_dataset.py     # 导出任务二训练数据包
│   ├── deap_task1/                  # 任务一预处理实现
│   └── deap_task2/                  # 任务二数据整理实现
└── tests/                           # 任务一契约测试
```

仓库不提交原始数据、清洗后的大文件、图片和 `data/task2/` 生成结果。

## 环境安装

在 `Task1-preprocess/` 下执行：

```bash
python -m pip install -e . pytest
```

## 数据放置

默认只认仓库内相对路径：

```text
Task1-preprocess/data/raw/DEAP/data_original/s01.bdf
Task1-preprocess/data/raw/DEAP/data_original/s02.bdf
...
Task1-preprocess/data/raw/DEAP/data_original/s32.bdf
Task1-preprocess/data/raw/DEAP/metadata_csv.zip
```

如果数据不放在默认位置，运行命令时用参数显式指定，不要把个人电脑路径写进代码或 README。

## 任务一：预处理 BDF

处理单个被试：

```bash
python src/run_task1.py --subject s01
```

处理前 2 个被试：

```bash
python src/run_task1.py --limit 2
```

数据放在其他位置时：

```bash
python src/run_task1.py --data-root "<你的DEAP原始BDF目录>" --subject s01
```

任务一输出：

| 路径 | 内容 |
|---|---|
| `data/interim/<subject>/channel_table.csv` | 通道名称和通道类型 |
| `data/interim/<subject>/trial_events.csv` | 40 个 trial 的事件切分表 |
| `data/interim/<subject>/artifact_log.json` | ICA 和肌电伪迹处理日志 |
| `data/processed/<subject>/epochs_clean.fif` | 清洗后的 60 秒刺激段 epoch |
| `data/processed/<subject>/trial_qc.csv` | 每个 trial 的质量检查表 |
| `output/figures/<subject>/*.png` | 报告用图 |
| `output/task1_run_summary.csv` | 本次运行汇总 |

## 任务二：导出建模数据

只导出 trial 标签：

```bash
python src/prepare_task2_labels.py
```

导出完整训练数据包：

```bash
python src/prepare_task2_dataset.py
```

只导出指定被试：

```bash
python src/prepare_task2_dataset.py --subjects s01 s02
```

元数据不在默认位置时：

```bash
python src/prepare_task2_dataset.py --metadata-source "<metadata_csv.zip或metadata_csv目录>"
```

任务二输出：

| 路径 | 内容 |
|---|---|
| `data/task2/labels/deap_trial_labels.csv` | 每个 trial 的 valence、arousal、dominance、liking、familiarity 和二分类标签 |
| `data/task2/npz/<subject>_task2_trials.npz` | 建模可读的 EEG 数组 `X` 和对应标签 |
| `data/task2/manifests/task2_training_index.csv` | 所有 trial 的索引、标签、质量信息和文件路径 |
| `data/task2/manifests/task2_subject_summary.csv` | 每个被试的导出摘要 |
| `data/task2/manifests/task2_dataset_stats.json` | 本次任务二数据导出统计 |

## 检查

```bash
python -m pytest
```
