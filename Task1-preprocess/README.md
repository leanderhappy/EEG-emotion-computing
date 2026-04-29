# DEAP 任务一预处理工程-王文龙

## 1. 这套代码做什么

这套工程只处理 DEAP 原始 `BDF` 数据，不使用官方预处理版 `dat`。

完整流程是：

1. 读取原始 `BDF`
2. 统一通道类型和通道表
3. 对 EEG 做 `4-45 Hz` 带通滤波
4. 重参考，再重采样到 `128 Hz`
5. 用 ICA 识别眼电伪迹，用肌电标注识别高肌电片段
6. 按视频 trial 切段
7. 用前 `3 s` 基线做校正，只保留后 `60 s` 刺激段
8. 导出 `epochs_clean.fif`、中间表格和结果图

一句大白话：就是把原始脑电从“能读”整理成“能直接给任务二和课程报告接着用”。

## 2. 环境安装

建议先确认本机 Python 在 `3.12` 左右，再执行：

```bash
python -m pip install -e . pytest
```

如果不想用 `-e .`，至少安装这些依赖：

```bash
python -m pip install mne scipy scikit-learn pandas matplotlib pytest
```

## 3. 数据怎么放

原始数据目录：

```text
D:\大三下_AIA\data\raw\DEAP\data_original
```

目录里应该直接能看到：

```text
s01.bdf
s02.bdf
...
s32.bdf
```

只接受原始 `BDF`。如果放进去的是官方预处理版 `dat`，这套代码就不符合课程要求。

## 4. 运行命令

### 跑一个被试

```bash
python src/run_task1.py --data-root "D:\大三下_AIA\data\raw\DEAP\data_original" --subject s01
```

### 跑前 2 个被试

```bash
python src/run_task1.py --data-root "D:\大三下_AIA\data\raw\DEAP\data_original" --limit 2
```

### 如果数据已经放在默认目录

```bash
python src/run_task1.py --subject s01
```

## 5. 输出目录速查

| 路径                             | 里面是什么           | 最适合谁看        |
| ------------------------------ | --------------- | ------------ |
| `data/interim/s01/`            | 中间结果表和伪迹日志      | 做方法说明的人      |
| `data/processed/s01/`          | 最终清洗后的 epoch 数据 | 做后续建模的人      |
| `output/figures/s01/`          | 课程报告直接可用的图      | 写报告和做 PPT 的人 |
| `output/task1_run_summary.csv` | 这次跑了哪些被试、输出落到哪里 | 总负责人         |

## 6. 课程要求和输出文件怎么一一对应

| 课程要求                 | 对应文件                                                                                                                                                               | 应该看什么                                      | 报告里怎么写                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | -------------------------------------------------- |
| 数据导入与格式转换            | `data/interim/s01/channel_table.csv`、`data/interim/s01/trial_events.csv`                                                                                           | 通道是否被分成 EEG、EOG、EMG、stim；trial 起点终点能不能正确解析 | 写“原始 BDF 成功导入，完成通道类型标准化，并从状态通道解析出 40 个视频 trial 事件” |
| 滤波：带通滤波（去除工频干扰）      | `output/figures/s01/psd_comparison.png`                                                                                                                            | `Before` 和 `After` 两条谱线变化                  | 写“滤波后低频漂移减弱，工频附近干扰下降，频谱更集中在有效脑电频段”                 |
| 伪迹去除：去除眼电、肌电等伪迹      | `output/figures/s01/ica_eog_scores.png`、`output/figures/s01/ica_eog_topomap.png`、`output/figures/s01/waveform_comparison.png`、`data/interim/s01/artifact_log.json` | 哪些 ICA 成分最像眼电；清洗前后波形是否更稳；肌电标注了多少秒          | 写“采用 ICA 识别眼电相关独立成分，并结合肌电标注记录高肌电片段，完成伪迹控制”         |
| 数据分段：以视频刺激为基准，切分实验时段 | `data/interim/s01/trial_events.csv`                                                                                                                                | 是否一共 40 个 trial；每个 trial 是否有基线起点、视频起点、视频终点 | 写“以视频刺激起点为基准完成 trial 切分，并保留基线段与刺激段对应关系”            |
| 基线校正：标准化信号           | `data/processed/s01/epochs_clean.fif`、`output/figures/s01/waveform_comparison.png`                                                                                 | 最终 epoch 是否统一成 `60 s`；清洗后起始漂移是否更小          | 写“使用 trial 前 3 秒基线做逐通道校正，输出统一长度的 60 秒刺激段数据”        |

## 7. 每个核心文件到底是什么

| 文件                                           | 含义                | 大白话理解                        |
| -------------------------------------------- | ----------------- | ---------------------------- |
| `data/interim/s01/channel_table.csv`         | 通道名和通道类型表         | 告诉你每根“线”到底是脑电、眼电、肌电还是状态码     |
| `data/interim/s01/trial_events.csv`          | 40 个 trial 的事件表   | 告诉你每个视频什么时候开始，前 3 秒基线从哪儿算    |
| `data/interim/s01/artifact_log.json`         | 伪迹处理日志            | 记录 ICA 排除了哪些成分、最像眼电的是谁、肌电有多少 |
| `data/processed/s01/epochs_clean.fif`        | 最终可直接分析的脑电 epoch  | 这是任务一最核心的成品                  |
| `data/processed/s01/trial_qc.csv`            | 每个 trial 的质量检查表   | 哪个 trial 和肌电脏片段重叠更多，一眼能看出来   |
| `output/figures/s01/psd_comparison.png`      | 滤波前后频谱图           | 看滤波是不是把脏频率压下去了               |
| `output/figures/s01/waveform_comparison.png` | 清洗前后波形图           | 看信号是否更平稳、突发尖峰是否减少            |
| `output/figures/s01/ica_eog_scores.png`      | ICA 成分和眼电通道的相关分数图 | 看哪些成分最像眨眼或眼动                 |
| `output/figures/s01/ica_eog_topomap.png`     | 最像眼电的 ICA 成分头皮分布图 | 看这些成分在头皮上的空间分布像不像眼部伪迹        |
| `output/task1_run_summary.csv`               | 本次批处理总表           | 方便总负责人检查哪些被试已经跑完             |

## 8. 写报告时，建议怎么展示

| 想展示的步骤  | 最推荐截图                                                                                                                         | 截图时重点圈哪里                                                           |
| ------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 导入与格式转换 | `data/interim/s01/channel_table.csv`、`data/interim/s01/trial_events.csv`                                                      | 圈出 EEG、EOG、EMG、stim 分类，以及 40 个 trial                               |
| 滤波      | `output/figures/s01/psd_comparison.png`                                                                                       | 圈出 `Before`、`After`，说明滤波后低频漂移和工频干扰下降                               |
| 伪迹去除    | `output/figures/s01/ica_eog_scores.png`、`output/figures/s01/ica_eog_topomap.png`、`output/figures/s01/waveform_comparison.png` | 圈出最像眼电的几个 component，以及清洗前后波形变化                                     |
| 分段      | `data/interim/s01/trial_events.csv`                                                                                           | 圈出 `baseline_start_sample`、`video_start_sample`、`video_end_sample` |
| 基线校正    | `output/figures/s01/waveform_comparison.png` 加 `epochs_clean.fif` 说明                                                          | 写明最终每段只保留 60 秒刺激期                                                  |

## 9. 看图手册：给写报告同学看的大白话版本

### 9.1 `psd_comparison.png`

- `Before`：滤波前的频谱。
- `After`：滤波后的频谱。
- 横轴是频率，纵轴是这个频率上的能量强弱。
- 如果 `After` 比 `Before` 更干净，说明滤波有效。
- 低频一大坨往下掉，表示慢漂移被压住了。
- `50 Hz` 附近如果更低，通常说明工频污染影响更小了。

### 9.2 `waveform_comparison.png`

- 每一行是一根代表通道，比如前额、中间、后脑。
- `Before` 是清洗前，`After` 是清洗后。
- 这张图不是看“完全一条直线”，而是看：
  - 基线有没有更靠近 0
  - 大尖峰有没有减少
  - 同一时间段里波形是不是没那么乱
- 太平反而可能有问题，正常脑电本来就会起伏。

### 9.3 `ica_eog_scores.png`

- 每条线对应一根眼电通道，也就是 `EXG1` 到 `EXG4`。
- 横轴 `component index` 是 ICA 分出来的第几个独立成分，可以理解成“拆出来的第几种隐藏信号来源”。
- 纵轴 `score` 表示这个成分和某根眼电通道有多像。
- 绝对值越大，越像眼电伪迹。
- 红色竖线如果出现，表示这个成分被加入了待剔除名单。
- 图上标出来的 `C18 / EXG1` 这种字样，意思是“第 18 个成分最像 EXG1 这根眼电通道”。

### 9.4 `ica_eog_topomap.png`

- 每个小头图对应一个最像眼电的 ICA 成分。
- 标题里的 `C18` 这种编号，就是 `component index`。
- 它不是原始第 18 根电极，也不是第 18 个 trial。
- 它只是 ICA 分解后得到的“第 18 个独立成分编号”。
- 头皮图颜色表示这个成分在不同电极上的权重大小。
- 如果前额区域特别强，常常说明它更像眨眼或眼动伪迹。

## 10. 结果核对清单

| 检查项     | 过线标准                       | 去哪里看                                                                             |
| ------- | -------------------------- | -------------------------------------------------------------------------------- |
| 能读原始数据  | 直接读 `BDF`，不是 `dat`         | 终端日志、`output/task1_run_summary.csv`                                              |
| trial 数 | 每个被试有 40 个 trial           | `data/interim/s01/trial_events.csv`                                              |
| 最终时长    | 每个最终 epoch 都是 60 秒刺激段      | `data/processed/s01/epochs_clean.fif`                                            |
| 滤波效果    | `Before/After` 频谱有明显差异     | `output/figures/s01/psd_comparison.png`                                          |
| 眼电处理    | 能找出最像眼电的 component         | `output/figures/s01/ica_eog_scores.png`、`output/figures/s01/ica_eog_topomap.png` |
| 肌电处理    | 有肌电重叠统计和日志                 | `data/interim/s01/artifact_log.json`、`data/processed/s01/trial_qc.csv`           |
| 输出能复用   | 任务二能直接接 `epochs_clean.fif` | `data/processed/s01/epochs_clean.fif`                                            |

## 11. 常见问题

| 问题                               | 原因               | 处理                            |
| -------------------------------- | ---------------- | ----------------------------- |
| 找不到 `.bdf`                       | 数据目录放错了          | 检查 `--data-root`              |
| `Missing expected DEAP channels` | 不是标准 DEAP 原始 BDF | 换回原始数据，不要硬凑                   |
| trial 不是 40 个                    | 事件码缺失或文件不完整      | 直接检查原始 BDF                    |
| 图没生成                             | 中间某一步报错了         | 先看终端，再看 `artifact_log.json`   |
| README 中文乱码                      | 打开方式不是 UTF-8     | 用 VS Code 或 Typora 以 UTF-8 打开 |

## 12. 当前推荐阅读顺序

1. 先看 `output/task1_run_summary.csv`
2. 再看 `data/interim/s01/trial_events.csv`
3. 然后看 `output/figures/s01/psd_comparison.png`
4. 接着看 `output/figures/s01/ica_eog_scores.png`
5. 再看 `output/figures/s01/ica_eog_topomap.png`
6. 最后看 `data/processed/s01/epochs_clean.fif`

这样看一圈，基本就能把任务一的“导入 -> 滤波 -> 去伪迹 -> 切段 -> 基线校正 -> 导出结果”完整讲清楚。
