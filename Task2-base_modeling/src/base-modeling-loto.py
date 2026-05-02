import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.signal import welch
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from pathlib import Path
import pickle
import argparse
import logging
import sys

def extract_deap_features(trial_data, sfreq=128):
    # 输入形状：(通道,时间)
    bands = [(4, 8), (8, 12), (12, 30), (30, 45)]  # theta, alpha, beta, gamma
    
    # 计算 PSD
    freqs, psd = welch(trial_data, fs=sfreq, nperseg=256, axis=1)
    
    features = []
    
    # # 频带功率特征
    # for low, high in bands:
    #     idx = np.logical_and(freqs >= low, freqs <= high)
    #     band_psd = np.mean(psd[:, idx], axis=1)
    #     features.append(band_psd)
    
    # 微分熵
    for low, high in bands:
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.mean(psd[:, idx], axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * band_power + 1e-8)
        features.append(de)

    # # 时域统计特征
    # for ch in range(len(trial_data)):
    #     data = trial_data[ch]
    #     features.append([np.mean(data), np.std(data), 
    #                      np.mean(np.diff(data)**2),   # 二阶差分均值 (活动性)
    #                      np.mean(np.abs(np.diff(data)))])  # 一阶差分绝对均值 (移动性)
    
    return np.concatenate([np.array(f).flatten() for f in features])

def load_data(data_root):
    '''
    加载数据
    Args:
        data_root: dat/npz数据所在目录
    Returns:
        X_all: 原始脑电数据，形状 (被试C, 实验B, 通道N, 时间T)
        y_all: 归一化到(1/9, 2/9, ...)的效价，形状(被试C, 实验B)
    '''
    # 选择其中的14个通道
    channels = [1,2,3,4,6,11,13,17,19,20,21,25,29,31]

    # 加载数据
    data_list = sorted(
        list(Path(data_root).glob("*.dat")) + 
        list(Path(data_root).glob("*.npz"))
    )
    X_all, y_all = [], []

    if len(data_list) == 0:
        raise FileNotFoundError(f"No .dat or .npz files found in {data_root}")
    
    for data_path in data_list:
        if data_path.suffix == ".dat":
            with open(data_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            X = data["data"]              # (40, 40, 8064)：实验 / 通道 / 时间    (B, N, T)
            y = data["labels"][:, 0]      # (40, 4): 效价、唤醒度、支配度、喜爱度

            # 减去基线值
            baseline = X[:, :, :384].mean(axis=-1, keepdims=True)
            X = X[:, :, 384:] - baseline
        else:
            data = np.load(data_path)
            X = data["X"]
            y = data["valence"]
        
        # 选取部分通道
        X = X[:, channels, :]

        # # 个体归一化
        # mu = np.mean(X, axis=(0, 2), keepdims=True)   # shape: (1, 40, 1)
        # sigma = np.std(X, axis=(0, 2), keepdims=True) # shape: (1, 40, 1)
        # X = (X - mu) / (sigma + 1e-8)

        feat = []
        for i in range(len(X)):
            feat.append(extract_deap_features(X[i]))
        
        X_all.append(feat)
        y_all.append(y / 9)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    # y_all_binary = (y_all > 5/9).astype(int)  # 二分类标签
    return X_all, y_all

def base_modeling_regression(X_train, y_train, model_type, use_gridsearch=True):
    # 使用默认参数
    if not use_gridsearch:
        if model_type == "svm":
            model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
        elif model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
            )
        elif model_type == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(128, 128),
                activation='relu',
                solver='adam',
                alpha=0.05,
                batch_size=64,
                learning_rate='adaptive',
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
            )
        
        model.fit(X_train, y_train)
        return model
    
    # 使用GridSearchCV
    if model_type == "svm":
        base_model = SVR()
        param_grid = {
            'kernel': ['rbf'],
            'C': [0.01, 0.1, 1.0],
            'gamma': ['scale'],
            'epsilon': [0.01, 0.1]
        }
        
    elif model_type == "rf":
        base_model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5],
            'max_features': ['sqrt']
        }
        
    elif model_type == "mlp":
        base_model = MLPRegressor(
            early_stopping=True,
            validation_fraction=0.1,
        )
        param_grid = {
            'hidden_layer_sizes': [(32, 32), (64, 64), (128, 128)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.05],
            'batch_size': [32, 64, 128],
            'learning_rate': ['adaptive']
        }
    
    cv = KFold(n_splits=3, shuffle=True)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='r2',
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=False
    )
    
    print("Begin grid search.")
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def base_modeling_classify(X_train, y_train, model_type, use_gridsearch=True, cv_splits=3, scoring='accuracy'):
    if not use_gridsearch:
        # 使用默认参数
        if model_type == "svm":
            model = SVC(kernel='rbf', C=1.0, gamma='scale')
        elif model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
            )
        elif model_type == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(32, 32),
                activation='relu',
                solver='adam',
                alpha=0.05,
                batch_size=64,
                learning_rate='adaptive',
                max_iter=200,
            )
        else:
            raise ValueError(f"不支持模型: {model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    # 使用GridSearchCV
    if model_type == "svm":
        base_model = SVC()
        param_grid = {
            'kernel': ['rbf'],
            'C': [0.01, 0.1, 1.0],
            'gamma': ['scale'],
        }
        
    elif model_type == "rf":
        base_model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5],
            'max_features': ['sqrt'],
            'bootstrap': [True]
        }
        
        
    elif model_type == "mlp":
        base_model = MLPClassifier(
            early_stopping=True,
            validation_fraction=0.1,
        )
        param_grid = {
            'hidden_layer_sizes': [(32, 32), (64, 64), (128, 128)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.05],
            'batch_size': [32, 64, 128],
            'learning_rate': ['adaptive']
        }
    else:
        raise ValueError(f"不支持模型: {model_type}")
    
    if len(np.unique(y_train)) > 2:  # 多分类
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True)
    else:  # 二分类
        cv = KFold(n_splits=cv_splits, shuffle=True)
    
    # 创建GridSearchCV对象
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
        refit=True,  # 使用最佳参数在整个训练集上重新训练
        error_score='raise'  # 参数组合出错时抛出异常
    )
    
    print("Begin grid search.")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# def base_modeling_regression(X_train, y_train, model_type):
#     if model_type == "svm":
#         model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
#     elif model_type == "rf":
#         model = RandomForestRegressor(
#             n_estimators=100,        # 树的数量
#             max_depth=10,            # 限制树深度
#             min_samples_split=5,     # 分裂最小样本数
#             min_samples_leaf=2,      # 叶节点最小样本数
#             max_features='sqrt',     # 每棵树考虑的特征数
#         )
#     elif model_type == "mlp":
#         model = MLPRegressor(
#             hidden_layer_sizes=(128, 128),
#             activation='relu',
#             solver='adam',
#             alpha=0.05,
#             batch_size=64,
#             learning_rate='adaptive',
#             max_iter=200,
#             early_stopping=True,
#             validation_fraction=0.1,
#         )

#     model.fit(X_train, y_train)
#     return model

# def base_modeling_classify(X_train, y_train, model_type):
#     if model_type == "svm":
#         model = SVC(kernel='rbf', C=1.0, gamma='scale')
#     elif model_type == "rf":
#         model = RandomForestClassifier(
#             n_estimators=100,        # 树的数量
#             max_depth=5,            # 限制树深度
#             min_samples_split=10,     # 分裂最小样本数
#             min_samples_leaf=5,      # 叶节点最小样本数
#             max_features='sqrt',     # 每棵树考虑的特征数
#         )
#     elif model_type == "mlp":
#         model = MLPClassifier(
#             hidden_layer_sizes=(32, 32),
#             activation='relu',
#             solver='adam',
#             alpha=0.05,
#             batch_size=64,
#             learning_rate='adaptive',
#             max_iter=200,
#         )
#     model.fit(X_train, y_train)
#     return model

def run_base_modeling_loto(
        data_root, output_root, 
        num_repeats=1, model_type="rf", task='binary', 
        use_gridsearch=True, regression=True, seed=42
    ):
    set_seed(seed)
    output_root = Path(output_root) / (task+"-loto") / (model_type+"-regression" if regression else model_type)
    output_root.mkdir(parents=True, exist_ok=True)
    print("Output root:", output_root)

    # 加载数据
    print("Loading data.")
    X_all, y_all = load_data(data_root)

    # 重复实验多次
    train_acc_all, test_acc_all = [], []
    test_pred_all, test_real_all = [], []

    for i in range(num_repeats):
        print(f"========== Repeat {i} ==========")
        test_pred, test_real = [], []         # 测试集上的预测类别与实际类别
        train_pred, train_real = [], []

        # 每个被试
        for X, y in zip(X_all, y_all):
            # LOTO, Leave-One-Trial-Out
            idx = np.random.choice(len(X))
            X_test, y_test = X[idx:idx+1], classify(y[idx:idx+1], task)
            X_train = np.delete(X, idx, axis=0)
            y_train = np.delete(y, idx, axis=0)             # 连续值
            y_train_category = classify(y_train, task)      # 离散类别
        
            # 标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 特征选择
            k = min(60, X.shape[1])  # 取较小值
            selector = SelectKBest(mutual_info_regression, k=k)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

            if regression:
                model = base_modeling_regression(X_train, y_train, model_type, use_gridsearch)
                y_train_pred = classify(model.predict(X_train), task)
                y_test_pred = classify(model.predict(X_test), task)
            else:
                model = base_modeling_classify(X_train, y_train_category, model_type, use_gridsearch)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
            
            train_pred.extend(y_train_pred)
            train_real.extend(y_train_category)
            test_pred.extend(y_test_pred)
            test_real.extend(y_test)

        train_acc = accuracy_score(train_pred, train_real)
        test_acc = accuracy_score(test_pred, test_real)
        print("train accuracy:", train_acc)
        print("test accuracy:", test_acc)
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        test_pred_all.extend(test_pred)
        test_real_all.extend(test_real)

    if task == "binary":
        category_labels = ["N", "P"]
    elif task == "threeclass":
        category_labels = ["N", "U", "P"]

    log_file = Path(output_root) / "loto-results.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("==========train_acc==========")
    logging.info(train_acc_all)
    logging.info("mean: %s", np.mean(train_acc_all))
    logging.info("==========test_acc==========")
    logging.info(test_acc_all)
    logging.info("mean: %s", np.mean(test_acc_all))
    logging.info("==========classification_report==========")
    logging.info(classification_report(test_real_all, test_pred_all, target_names=category_labels))

    fig_file = Path(output_root) / "loto-confusion_matrix.png"
    cm = confusion_matrix(test_real_all, test_pred_all)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=category_labels,
                yticklabels=category_labels)
    plt.xlabel('pred')
    plt.ylabel('real')
    plt.title(f'Confusion Matrix')
    plt.tight_layout()
    plt.savefig(fig_file)
    plt.show()
    print("logs saved to:", log_file)
    print("figure saved to:", fig_file)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def classify(data, method):
    if method == "binary":
        return np.array(data > 5/9)
    elif method == "threeclass":
        return np.digitize(data, bins=[4/9, 6/9])
    

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    print("project root:", project_root)

    parser = argparse.ArgumentParser("DEAP task2 baseline modeling with LOTO")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=project_root / "Task1-preprocess" / "data" / "task2" / "npz",
        help="directory of data files"
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=project_root / "Task2-base_modeling" / "output",
        help="directory for output files"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        choices=["svm", "rf", "mlp"],
        help="choose a model from SVM, RandomForest, MLP"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["binary", "threeclass"],
        help="binary: valence high/low; threeclass: negative/neutral/positive"
    )
    parser.add_argument(
        "--regression", 
        action="store_true",
        default=False,
        help='是否采用回归模式训练模型'
    )
    parser.add_argument(
        "--no_gridsearch", 
        action="store_true",
        default=False,
        help='是否采用网格搜索超参数，若加入此项，则使用预设超参数'
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=10,
        help="重复实验次数"
    )
    args = parser.parse_args()

    run_base_modeling_loto(
        data_root = args.data_root,
        output_root = args.output_root,
        num_repeats = args.num_repeats, 
        model_type = args.model_type,
        task = args.task,
        regression = args.regression,
        use_gridsearch=not args.no_gridsearch,
        seed = 1
    )
