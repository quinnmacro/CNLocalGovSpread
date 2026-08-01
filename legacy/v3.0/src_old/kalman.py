"""
卡尔曼滤波器模块 - 从噪音中提取真实信号

技术实现:
- 使用 statsmodels 的 SARIMAX，令 order=(0,1,0) 就是 Local Level Model
- 如果 Kalman 优化失败（在极短样本或极端波动下可能发生），fallback 到 60 日移动均值
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class KalmanSignalExtractor:
    """
    卡尔曼滤波器 - 从噪音中提取真实信号

    技术实现:
    - 使用 statsmodels 的 SARIMAX，令 order=(0,1,0) 就是 Local Level Model
    - 如果 Kalman 优化失败（在极短样本或极端波动下可能发生），fallback 到 60 日移动均值
    """

    def __init__(self, spread_series):
        self.spread = spread_series
        self.smoothed_state = None
        self.success = False

    def fit(self, fallback_window=60):
        """
        拟合卡尔曼滤波器

        参数:
        - fallback_window: 失败时滚动均值窗口 (默认60, 可由参数校准提供)
        """
        """
        拟合 Local Level Model (随机游走 + 噪音)

        数学表达:
        - μ_t = μ_{t-1} + η_t        (状态方程: 随机游走)
        - y_t = μ_t + ε_t            (观测方程: 真实值 + 噪音)

        其中 η_t ~ N(0, σ_η^2), ε_t ~ N(0, σ_ε^2) 由 MLE 估计
        """
        print("\n" + "="*60)
        print("拟合卡尔曼滤波器 (Local Level Model)")
        print("="*60)

        try:
            # SARIMAX(order=(0,1,0)) 等价于:
            # - 0 个 AR 项
            # - 1 阶差分（这会自动构建随机游走状态空间）
            # - 0 个 MA 项
            model = SARIMAX(self.spread, order=(0, 1, 0), trend=None)

            # method='bfgs' 通常比默认的 'lbfgs' 更稳定
            # maxiter=500 给优化器足够多的迭代次数
            result = model.fit(disp=False, method='bfgs', maxiter=500)

            # get_smoothed_decomposition() 是 Kalman Smoother 的输出
            # 它用全部样本信息（包括未来）来估计每个时点的状态 → 比 Filter 更准确
            smoothed_array = result.smoothed_state[0]  # 取第一个状态变量
            # 转换为 pandas Series，保持原始索引
            self.smoothed_state = pd.Series(smoothed_array, index=self.spread.index)
            self.success = True

            print("✓ Kalman Filter 拟合成功")
            # P0修复: SARIMAX(order=(0,1,0))的sigma2是状态转移误差方差(σ_η²)，非观测误差(σ_ε²)
            # 参考: Hamilton (1994) Time Series Analysis, Chapter 13
            print(f"  状态转移噪音标准差 σ_η ≈ {np.sqrt(result.params['sigma2']):.2f} bps")
            # 注: 真实观测噪音需要检查 result.filter_results.obs_cov

        except Exception as e:
            print(f"⚠️  Kalman Filter 优化失败: {str(e)}")
            print("   启用 Fallback: 使用 60 日滚动均值代替")

            # Fallback: 简单移动平均
            # 60 天窗口的选择：对应一个季度的财报周期
            # 太短（如 20 天）会追踪噪音，太长（如 120 天）反应太慢
            self.smoothed_state = self.spread.rolling(window=fallback_window, min_periods=1).mean()
            self.success = False

        return self.smoothed_state

    def get_signal_deviation(self):
        """
        计算当前利差偏离 Kalman 趋势的程度（标准化）

        这个指标可以用来构建均值回归策略:
        - deviation > +1.5σ → 利差高估，做空信号（预期收窄）
        - deviation < -1.5σ → 利差低估，做多信号（预期扩大）

        P0修复: 使用标准化创新(standardized innovation)而非残差样本标准差
        标准化创新 = (观测值 - 预测值) / 预测误差标准差
        这是Kalman滤波理论中正确的标准化方法，提供了时变的不确定性估计
        """
        if self.smoothed_state is None:
            raise ValueError("请先调用 fit()")

        try:
            # 尝试获取Kalman滤波器的标准化创新
            # 标准化创新 = one-step-ahead prediction error / sqrt(prediction error variance)
            model = SARIMAX(self.spread, order=(0, 1, 0), trend=None)
            result = model.fit(disp=False, method='bfgs', maxiter=500)

            # 获取标准化创新（时变的Z-score）
            # standardized_forecasts_error 已经被预测误差标准差标准化
            innovation = result.filter_results.standardized_forecasts_error[0]

            # 对齐索引（可能比原始数据少一个，因为差分）
            if len(innovation) == len(self.spread) - 1:
                # 差分模型，第一个观测值没有预测误差
                normalized_deviation = pd.Series(innovation, index=self.spread.index[1:])
                # 第一个值用残差方法填充
                first_dev = (self.spread.iloc[0] - self.smoothed_state.iloc[0]) / (self.spread - self.smoothed_state).std()
                normalized_deviation = pd.concat([
                    pd.Series([first_dev], index=[self.spread.index[0]]),
                    normalized_deviation
                ])
            else:
                normalized_deviation = pd.Series(innovation, index=self.spread.index)

            return normalized_deviation

        except Exception:
            # Fallback: 使用残差标准差（保留原有逻辑作为后备）
            deviation = self.spread - self.smoothed_state
            std = deviation.std()

            # 添加保护，防止除以接近零的标准差
            if std < 1e-6:
                std = 1e-6

            normalized_deviation = deviation / std

            return normalized_deviation
