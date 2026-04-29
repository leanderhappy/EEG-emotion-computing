# 数据目录

这里放任务一预处理用到的数据和中间结果。

目录约定：

```text
data/
├── raw/DEAP/data_original/   # DEAP 原始 BDF 文件，文件名如 s01.bdf
├── interim/                  # 通道表、trial 事件表、伪迹日志等中间结果
└── processed/                # 清洗后的 epochs_clean.fif 和 trial_qc.csv
```

只提交目录说明，不提交本地生成的大体积数据文件。
