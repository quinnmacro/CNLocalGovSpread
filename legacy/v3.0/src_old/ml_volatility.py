"""
ML波动率模型对比模块 - 机器学习方法与GARCH锦标赛竞争

技术实现:
- Random Forest: 滚动窗口特征提取 + 非参数预测
- XGBoost: 梯度提升树，捕捉非线性波动率动态
- LSTM: 长短期记忆网络，学习序列依赖关系

设计理念:
- 所有ML模型与GARCH锦标赛使用统一评估框架（AIC/BIC + RMSE/MAE）
- ML模型输入为标准化特征矩阵，输出为波动率预测
- 使用Mock数据兼容模式，无需外部数据源即可运行
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class MLVolatilityModeler:
    """
    ML波动率模型对比类 - Random Forest / XGBoost / LSTM 竞争GARCH锦标赛

    关键设计:
    - 统一特征工程: 滚动窗口统计特征（均值、方差、偏度、峰度等）
    - 统一评估指标: AIC/BIC（与GARCH可比） + RMSE/MAE（ML标准）
    - 自动超参数选择: 网格搜索或默认最优参数
    """

    def __init__(self, returns, window_size=20):
        """
        参数:
        - returns: pd.Series, 利差变化序列
        - window_size: int, 滚动窗口大小（默认20日）
        """
        self.returns = returns
        self.window_size = window_size
        self.models = {}
        self.ic_scores = {}
        self.predictions = {}
        self.feature_names = []

    def _build_features(self):
        """
        构建滚动窗口特征矩阵

        特征列表:
        1. 滚动均值 (rolling_mean)
        2. 滚动方差 (rolling_var)
        3. 滚动偏度 (rolling_skew)
        4. 滚动峰度 (rolling_kurt)
        5. 滚动最大绝对值 (rolling_max_abs)
        6. 滚动范围 (rolling_range)
        7. 当前收益率 (current_return)
        8. 前一期收益率平方 (lag1_sq)
        9. EWMA方差 (ewma_var)
        """
        data = self.returns.values
        n = len(data)
        w = self.window_size

        # 需要至少 window_size + 1 个数据点
        if n < w + 1:
            raise ValueError(f"数据长度 {n} 不足以构建 {w} 日滚动窗口特征")

        features = []
        feature_names = [
            'rolling_mean', 'rolling_var', 'rolling_skew', 'rolling_kurt',
            'rolling_max_abs', 'rolling_range', 'current_return',
            'lag1_sq', 'ewma_var'
        ]
        self.feature_names = feature_names

        # EWMA方差预计算
        ewma_var = np.zeros(n)
        ewma_lambda = 0.94
        ewma_var[0] = data[0] ** 2
        for t in range(1, n):
            ewma_var[t] = ewma_lambda * ewma_var[t-1] + (1 - ewma_lambda) * data[t-1] ** 2

        for t in range(w, n):
            window_data = data[t-w:t]

            rolling_mean = np.mean(window_data)
            rolling_var = np.var(window_data)
            rolling_skew = self._skewness(window_data)
            rolling_kurt = self._kurtosis(window_data)
            rolling_max_abs = np.max(np.abs(window_data))
            rolling_range = np.max(window_data) - np.min(window_data)
            current_return = data[t]
            lag1_sq = data[t-1] ** 2

            features.append([
                rolling_mean, rolling_var, rolling_skew, rolling_kurt,
                rolling_max_abs, rolling_range, current_return,
                lag1_sq, ewma_var[t-1]
            ])

        X = np.array(features)

        # 目标变量: 下一步的绝对收益率（作为波动率代理）
        # y[t] = |return[t+1]| 对应特征窗口 data[t-w:t+1]
        # 但我们的特征从 t=w 开始，目标从 y[w] = |data[w]| 不对
        # 正确: 特征在 t 时刻构建（用 data[t-w:t]），目标是 |data[t]|²（条件方差）
        # 使用 |data[t]|² 作为波动率代理（与GARCH的条件方差可比）
        y = np.zeros(len(X))
        for i in range(len(X)):
            t_idx = w + i
            y[i] = data[t_idx] ** 2  # 条件方差代理

        return X, y, feature_names

    def _skewness(self, x):
        """计算偏度"""
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return np.mean(((x - mean) / std) ** 3)

    def _kurtosis(self, x):
        """计算超额峰度"""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return np.mean(((x - mean) / std) ** 4) - 3

    def fit_random_forest(self, n_estimators=200, max_depth=10):
        """
        拟合Random Forest波动率预测模型

        参数:
        - n_estimators: 树的数量，默认200（足够稳定）
        - max_depth: 最大深度，默认10（防止过拟合）
        """
        print("\n[ML-1/3] 拟合 Random Forest 波动率模型...")

        X, y, feature_names = self._build_features()

        # 时间序列交叉验证: 前70%训练，后30%测试
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        # 评估指标
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        # AIC/BIC计算（与GARCH可比）
        # 使用测试集残差的对数似然
        residuals = y_test - y_pred_test
        n_test = len(y_test)
        sigma2 = np.var(residuals)
        if sigma2 < 1e-10:
            sigma2 = 1e-10

        # 正态分布对数似然
        log_likelihood = -n_test/2 * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)

        # Random Forest参数数（近似为有效树节点数）
        # 使用 n_estimators * 平均叶子数 作为参数数估计
        n_params = n_estimators * max_depth  # 简化估计
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_test) * n_params - 2 * log_likelihood

        self.models['RF'] = model
        self.ic_scores['RF'] = {
            'AIC': aic, 'BIC': bic,
            'RMSE': rmse_test, 'MAE': mae_test,
            'converged': True, 'n_params': n_params
        }

        # 存储完整预测序列
        full_pred = model.predict(X)
        pred_series = pd.Series(
            np.sqrt(full_pred),
            index=self.returns.index[self.window_size:]
        )
        self.predictions['RF'] = pred_series

        print(f"   RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
        print(f"   AIC={aic:.2f}, BIC={bic:.2f}")
        print(f"   特征重要性 Top-3: {self._get_top_features(model, feature_names)}")

        return pred_series

    def fit_xgboost(self, n_estimators=200, max_depth=6, learning_rate=0.05):
        """
        拟合XGBoost波动率预测模型

        参数:
        - n_estimators: boosting轮数
        - max_depth: 树深度
        - learning_rate: 学习率
        """
        print("\n[ML-2/3] 拟合 XGBoost 波动率模型...")

        try:
            from xgboost import XGBRegressor
        except ImportError:
            print("   ⚠️ XGBoost未安装，跳过此模型")
            print("   安装方法: pip install xgboost")
            self.ic_scores['XGBoost'] = {'AIC': np.inf, 'BIC': np.inf}
            return None

        X, y, feature_names = self._build_features()

        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        # AIC/BIC
        residuals = y_test - y_pred_test
        n_test = len(y_test)
        sigma2 = np.var(residuals)
        if sigma2 < 1e-10:
            sigma2 = 1e-10

        log_likelihood = -n_test/2 * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)

        n_params = n_estimators * max_depth
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_test) * n_params - 2 * log_likelihood

        self.models['XGBoost'] = model
        self.ic_scores['XGBoost'] = {
            'AIC': aic, 'BIC': bic,
            'RMSE': rmse_test, 'MAE': mae_test,
            'converged': True, 'n_params': n_params
        }

        full_pred = model.predict(X)
        pred_series = pd.Series(
            np.sqrt(full_pred),
            index=self.returns.index[self.window_size:]
        )
        self.predictions['XGBoost'] = pred_series

        print(f"   RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
        print(f"   AIC={aic:.2f}, BIC={bic:.2f}")
        print(f"   特征重要性 Top-3: {self._get_top_features(model, feature_names)}")

        return pred_series

    def fit_lstm(self, epochs=50, batch_size=32, sequence_length=20):
        """
        拟合LSTM波动率预测模型

        参数:
        - epochs: 训练轮数
        - batch_size: 批次大小
        - sequence_length: LSTM输入序列长度
        """
        print("\n[ML-3/3] 拟合 LSTM 波动率模型...")

        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM as LSTMLayer, Dense, Dropout
            # 设置TF警告级别
            tf.get_logger().setLevel('ERROR')
        except ImportError:
            print("   ⚠️ TensorFlow未安装，跳过LSTM模型")
            print("   安装方法: pip install tensorflow")
            self.ic_scores['LSTM'] = {'AIC': np.inf, 'BIC': np.inf}
            return None

        data = self.returns.values
        seq_len = sequence_length
        n = len(data)

        if n < seq_len + 50:
            print(f"   ⚠️ 数据长度不足 ({n} < {seq_len + 50})，跳过LSTM")
            self.ic_scores['LSTM'] = {'AIC': np.inf, 'BIC': np.inf}
            return None

        # 构建LSTM输入序列
        # X: (samples, timesteps, features) - 每个样本是seq_len个收益率
        # y: 下一步收益率的绝对值平方（波动率代理）
        X_lstm = []
        y_lstm = []
        for i in range(seq_len, n - 1):
            X_lstm.append(data[i-seq_len:i])
            y_lstm.append(data[i] ** 2)

        X_lstm = np.array(X_lstm).reshape(-1, seq_len, 1)
        y_lstm = np.array(y_lstm)

        # 标准化输入
        X_mean = np.mean(X_lstm)
        X_std = np.std(X_lstm)
        if X_std < 1e-10:
            X_std = 1e-10
        X_lstm_norm = (X_lstm - X_mean) / X_std

        # 时间序列分割
        split_idx = int(len(X_lstm_norm) * 0.7)
        X_train, X_test = X_lstm_norm[:split_idx], X_lstm_norm[split_idx:]
        y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        # 构建LSTM模型
        model = Sequential([
            LSTMLayer(64, input_shape=(seq_len, 1), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # 训练
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            shuffle=False  # 时间序列数据不应shuffle
        )

        y_pred_test = model.predict(X_test, verbose=0).flatten()

        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)

        # AIC/BIC
        residuals = y_test - y_pred_test
        n_test = len(y_test)
        sigma2 = np.var(residuals)
        if sigma2 < 1e-10:
            sigma2 = 1e-10

        log_likelihood = -n_test/2 * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)

        # LSTM参数数估计
        # LSTM(64): 4 * (64 * (1 + 64) + 64) = 4 * (64*65 + 64) = 4*4224 = 16896
        # Dense(32): 64*32 + 32 = 2080
        # Dense(1): 32*1 + 1 = 33
        n_params = 16896 + 2080 + 33  # ~19009
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_test) * n_params - 2 * log_likelihood

        self.models['LSTM'] = model
        self.ic_scores['LSTM'] = {
            'AIC': aic, 'BIC': bic,
            'RMSE': rmse_test, 'MAE': mae_test,
            'converged': True, 'n_params': n_params
        }

        # 全序列预测
        full_pred = model.predict(X_lstm_norm, verbose=0).flatten()
        pred_series = pd.Series(
            np.sqrt(np.maximum(full_pred, 0)),
            index=self.returns.index[seq_len+1:seq_len+1+len(full_pred)]
        )
        self.predictions['LSTM'] = pred_series

        print(f"   RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
        print(f"   AIC={aic:.2f}, BIC={bic:.2f}")
        print(f"   参数数: {n_params}")
        print(f"   训练轮数: {epochs}, 最终loss: {history.history['loss'][-1]:.6f}")

        return pred_series

    def run_ml_tournament(self):
        """
        执行ML模型锦标赛 - 与GARCH锦标赛对比

        返回:
        - dict: 所有模型的AIC/BIC/RMSE/MAE对比表
        """
        print("\n" + "="*60)
        print("ML波动率模型锦标赛")
        print("="*60)

        self.fit_random_forest()
        self.fit_xgboost()
        self.fit_lstm()

        # 生成对比表
        comparison = {}
        for name, scores in self.ic_scores.items():
            comparison[name] = {
                'AIC': scores['AIC'],
                'BIC': scores['BIC'],
                'RMSE': scores.get('RMSE', np.nan),
                'MAE': scores.get('MAE', np.nan)
            }

        # 选出ML Winner（按RMSE，更适合ML模型评估）
        valid_models = {k: v for k, v in comparison.items() if v['RMSE'] is not np.nan}
        if valid_models:
            ml_winner = min(valid_models, key=lambda x: valid_models[x]['RMSE'])
            print(f"\nML锦标赛获胜者: {ml_winner} (RMSE={valid_models[ml_winner]['RMSE']:.4f})")
        else:
            ml_winner = None
            print("\n⚠️ 无有效ML模型结果")

        print("="*60)

        return comparison, ml_winner

    def compare_with_garch(self, garch_ic_scores):
        """
        与GARCH锦标赛结果对比

        参数:
        - garch_ic_scores: VolatilityModeler.ic_scores 字典

        返回:
        - pd.DataFrame: 全模型对比表
        """
        all_scores = {}

        # 添加GARCH模型
        for name, scores in garch_ic_scores.items():
            all_scores[name] = {
                'AIC': scores['AIC'],
                'BIC': scores['BIC'],
                'RMSE': np.nan,  # GARCH没有RMSE
                'MAE': np.nan,
                'type': '计量经济学'
            }

        # 添加ML模型
        for name, scores in self.ic_scores.items():
            all_scores[name] = {
                'AIC': scores['AIC'],
                'BIC': scores['BIC'],
                'RMSE': scores.get('RMSE', np.nan),
                'MAE': scores.get('MAE', np.nan),
                'type': '机器学习'
            }

        # 综合排序（按AIC）
        sorted_models = sorted(all_scores.items(), key=lambda x: x[1]['AIC'])

        print("\n" + "="*60)
        print("全模型对比 (GARCH vs ML)")
        print("="*60)
        print(f"{'模型':<15} {'类型':<12} {'AIC':<12} {'BIC':<12} {'RMSE':<12} {'MAE':<12}")
        print("-" * 75)
        for name, scores in sorted_models:
            rmse_str = f"{scores['RMSE']:.4f}" if not np.isnan(scores['RMSE']) else 'N/A'
            mae_str = f"{scores['MAE']:.4f}" if not np.isnan(scores['MAE']) else 'N/A'
            print(f"{name:<15} {scores['type']:<12} {scores['AIC']:<12.2f} {scores['BIC']:<12.2f} {rmse_str:<12} {mae_str:<12}")

        overall_winner = sorted_models[0][0]
        print(f"\n综合获胜者 (AIC): {overall_winner}")
        print("="*60)

        df = pd.DataFrame(all_scores).T
        df.index.name = 'model'
        return df, overall_winner

    def _get_top_features(self, model, feature_names, top_n=3):
        """获取特征重要性Top-N"""
        try:
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            return [(feature_names[i], importances[i]) for i in top_indices]
        except (AttributeError, Exception):
            return []
