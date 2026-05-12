# DEAP 任务二：情感计算基础建模

本目录是课程任务二的独立工程：使用任务一预处理后的数据（或官方预处理数据`data_preprocessed_python`）进行情感效价分类，支持二分类（积极情感/消极情感）与三分类（积极/中性/消极）

## 目录结构

```
Task2-base_modeling/
├── data/
├── output/						# 输出结果
│   ├── official-preprocess/    	# 使用官方预处理数据分类结果
│   │   ├── binary-loto/      			# 二分类结果（LOTO划分方式）
│   │   ├── binary-random/      		# 二分类结果（random划分方式）
│   │   ├── threeclass-loto/      		# 三分类结果（LOTO划分方式）
│   │   └── threeclass-random/          # 三分类结果（random划分方式）
│   └── task1-preprocess/      		# 使用任务一预处理数据分类结果（目录结构与official-preprocess相同）
└── src/
    ├── base-modeling-merged.py     # 分类程序
    └── run_batch_tests.py			# 批量测试脚本
```

## 代码运行

### 1 base-modeling-merged.py

执行特定参数组合下的测试

```
python base-modeling-merged.py [--data_root DATA_ROOT] [--output_root OUTPUT_ROOT] [--model_type {svm,rf,mlp}] [--task {binary,threeclass}] [--regression] [--num_repeats NUM_REPEATS]
  --data_root DATA_ROOT			输入数据文件来源：official, task1, 或数据文件所在目录，支持自动处理.npz和.dat
  --output_root OUTPUT_ROOT		输出目录，默认为Task2-base_modeling\output
  --model_type {svm,rf,mlp}		模型类型（分别对应SVM, RandomForest, MLP）
  --task {binary,threeclass}	二分类 / 三分类
  --split {random,loto}			数据集划分方式
  --regression          		是否采用回归模式训练模型
  --no_gridsearch				是否采用网格搜索超参数，若加入此项，则使用预设超参数
  --num_repeats NUM_REPEATS		重复实验次数N，默认10
```

### 2 run_batch_tests.py

自动遍历`base-modeling-merged.py`每种选项组合情况，在所有组合上测试

```
python run_batch_tests.py
```

## 数据集划分方式

### 1 LOTO：Leave-One-Trial-Out

数据划分方式：对每个被试训练一个模型，从每个被试的40次实验中随机抽取一个作为测试集，剩下的39个作为训练集，整个实验重复N次。伪代码如下：

``` python
for repeat in range(N):			# 重复N次实验
    for candidate in range(32):	# 32名被试
        随机选择实验编号i (i=0,1,2,...,39)
        测试集 = candidate的第i次实验
        训练集 = candidate除第i次实验外的39次实验
        使用训练集训练模型
        使用测试集测试模型
结果汇总
```

### 2 Random

将所有被试全部实验批次数据混合在一起，随机划分训练集和测试集

## 是否采用回归模式（--regression）

- 如果采用回归模式训练，则加上`--regression`选项。在训练的时候直接用评分作为标签，预测时再根据模型输出分数判断它预测的类别（有时效果会好一点点）
- 如果不采用回归模式，则不加入该选项。在训练前就把标签分类为离散类别，直接使用离散类别标签训练

## 是否使用官方预处理数据集训练

若希望使用官方预处理数据集训练，则将`--data_root`设置为`official`。

（实测使用官方预处理数据集训练，测试集似乎准确率更高？）

## 输出文件格式

`base-modeling-merged.py`输出文件为`Task2-base_modeling\output\{预处理数据来源}\{task}-{split}\{model_type}[-regression]`：

- `confusion_matrix.png`：混淆矩阵图像
- `results.txt`：日志输出，包括训练/测试准确率、精确度、召回率、 f1-score等
