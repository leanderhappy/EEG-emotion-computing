import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
# import keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from pathlib import Path
import pickle

def extract_deap_features(trial_data, sfreq=128):
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
    # for ch in range(len(trail_data)):
    #     data = trial_data[ch]
    #     features.append([np.mean(data), np.std(data), 
    #                      np.mean(np.diff(data)**2),   # 二阶差分均值 (活动性)
    #                      np.mean(np.abs(np.diff(data)))])  # 一阶差分绝对均值 (移动性)
    
    return np.concatenate([np.array(f).flatten() for f in features])

def create_lstm_model(timesteps, n_features):
    model = Sequential()

    # model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(timesteps, n_features), return_sequences=True))
    # model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(timesteps, n_features)))
    model.add(LSTM(32, input_shape=(timesteps, n_features)))
    # model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    # model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

def run_baseline_modeling(
    dat_root, output_root, method="lstm",
    task='binary', split_mode='random',
    test_size=0.2, random_state=42, **kwargs
):
    # 选择其中的14个通道
    channels = [1,2,3,4,6,11,13,17,19,20,21,25,29,31]

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 加载数据
    dat_list = sorted(Path(dat_root).glob("*.dat"))
    X_all, y_all = [], []

    for dat_path in dat_list:
        with open(dat_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        X = data["data"]        # (40, 40, 8064)：实验 / 通道 / 时间    (B, N, T)
        y = data["labels"]      # (40, 4): 效价、唤醒度、支配度、喜爱度
        
        # 选取部分通道
        X = X[:, channels, :]

        # 个体归一化
        mu = np.mean(X, axis=(0, 2), keepdims=True)   # shape: (1, 40, 1)
        sigma = np.std(X, axis=(0, 2), keepdims=True) # shape: (1, 40, 1)
        X = (X - mu) / (sigma + 1e-8)

        # 减去基线值
        baseline = X[:, :, :384].mean(axis=-1, keepdims=True)
        X = X[:, :, 384:] - baseline

        for i in range(len(X)):
            if method == "lstm":
                feat = X[i]
            else:
                feat = extract_deap_features(X[i])
            X_all.append(feat)
            y_all.append(y[i, 0] / 9)

    X_all = np.array(X_all)     # (1280, 14, 7680)
    y_all = np.array(y_all)
    # y_all_binary = (y_all > 5/9).astype(int)  # 二分类标签

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )
    # X_train: (1024, 14, 7680)

    if method == "lstm":
        B, N, T = X_train.shape
        X_train = X_train.transpose(0,2,1).reshape(B, T//60, 60, N).mean(axis=2)
        B, N, T = X_test.shape
        X_test = X_test.transpose(0,2,1).reshape(B, T//60, 60, N).mean(axis=2)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]

        model = KerasRegressor(
            model=create_lstm_model,
            timesteps=X_train.shape[1],
            n_features=X_train.shape[2],
            epochs=50,
            batch_size=32,
            validation_split=0.2,  # 使用验证集
            callbacks=callbacks,
            verbose=1
        )
    else:
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 特征选择
        selector = SelectKBest(mutual_info_regression, k=60)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        if method == "svm":
            model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
        elif method == "rf":
            model = RandomForestRegressor(
                n_estimators=100,        # 树的数量
                max_depth=10,            # 限制树深度
                min_samples_split=5,     # 分裂最小样本数
                min_samples_leaf=2,      # 叶节点最小样本数
                max_features='sqrt',     # 每棵树考虑的特征数
                random_state=random_state
            )

            # model = RandomForestRegressor(
            #     n_estimators=100,        # 树的数量
            #     max_depth=5,            # 限制树深度
            #     min_samples_split=10,     # 分裂最小样本数
            #     min_samples_leaf=5,      # 叶节点最小样本数
            #     max_features='sqrt',     # 每棵树考虑的特征数
            #     random_state=random_state
            # )
        elif method == "mlp":
            # model = MLPRegressor(
            #     hidden_layer_sizes=(64, 64),
            #     activation='relu',
            #     solver='adam',
            #     alpha=0.05,
            #     batch_size=64,
            #     learning_rate='adaptive',
            #     max_iter=200,
            #     random_state=random_state
            # )

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
                random_state=random_state
            )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    y_pred_binary = y_pred > 5/9
    y_test_binary = y_test > 5/9
    y_pred_train_binary = y_pred_train > 5/9
    y_train_binary = y_train > 5/9

    print("=== 回归评估指标 ===")
    print("MAE (平均绝对误差): {:.4f}".format(mean_absolute_error(y_test, y_pred)))
    print("MSE (均方误差): {:.4f}".format(mean_squared_error(y_test, y_pred)))
    print("RMSE (均方根误差): {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("R² Score (决定系数): {:.4f}".format(r2_score(y_test, y_pred)))

    print("\n=== 模型评估 ===")
    print(f"训练集准确率: {accuracy_score(y_train_binary, y_pred_train_binary):.4f}")
    print(f"测试集准确率: {accuracy_score(y_test_binary, y_pred_binary):.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test_binary, y_pred_binary, target_names=['消极', '积极']))

    cm = confusion_matrix(y_test_binary, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['N', 'P'],
                yticklabels=['N', 'P'])
    plt.xlabel('pred')
    plt.ylabel('real')
    plt.title(f'Confusion Matrix ({method})')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_baseline_modeling("./data_preprocessed_python", "./output", method="rf")