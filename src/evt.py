"""
极值理论风险分析模块 - 量化尾部风险

方法: Peaks Over Threshold (POT)
分布: Generalized Pareto Distribution (GPD)
"""

import numpy as np
from scipy import stats


class EVTRiskAnalyzer:
    """
    极值理论风险分析器 - 量化尾部风险

    方法: Peaks Over Threshold (POT)
    分布: Generalized Pareto Distribution (GPD)
    """

    def __init__(self, returns, threshold_percentile=0.95, confidence=0.99):
        self.returns = returns
        self.threshold_percentile = threshold_percentile
        self.confidence = confidence
        self.threshold = None
        self.gpd_params = None
        self.var = None

    def fit_gpd(self):
        """
        拟合 GPD 到负尾部（损失）

        技术细节:
        - 我们关注利差扩大（正收益）的尾部风险
        - 选择 95% 分位数作为阈值是经验法则（McNeil & Frey 2000）
        - 太低（如 90%）会引入非极值数据，太高（如 99%）样本太少
        """
        print("\n" + "="*60)
        print("拟合极值理论 (EVT) 模型")
        print("="*60)

        # 定义阈值
        self.threshold = self.returns.quantile(self.threshold_percentile)

        # 提取超过阈值的极值（exceedances）
        exceedances = self.returns[self.returns > self.threshold] - self.threshold

        if len(exceedances) < 10:
            print(f"⚠️  警告: 超过阈值的样本太少 ({len(exceedances)} 个)，EVT 估计可能不稳定")

        print(f"阈值 u = {self.threshold:.2f} bps (第 {self.threshold_percentile*100:.0f} 百分位)")
        print(f"超过阈值的极值样本: {len(exceedances)} 个")

        try:
            # 拟合 GPD: 用 MLE 估计形状参数 ξ 和尺度参数 σ
            # ξ > 0: 重尾分布（典型的金融市场）
            # ξ = 0: 指数分布
            # ξ < 0: 轻尾分布（有上界，罕见）
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)  # floc=0 固定位置参数为 0

            self.gpd_params = {'shape': shape, 'scale': scale}

            print(f"\nGPD 参数估计:")
            print(f"  形状参数 ξ = {shape:.4f}")
            print(f"  尺度参数 σ = {scale:.4f}")

            if shape > 0.5:
                print(f"  ⚠️  ξ > 0.5 表示极端重尾，方差可能不存在！")
            elif shape > 0:
                print(f"  ✓ ξ > 0 确认重尾特征（符合金融数据）")

        except Exception as e:
            print(f"✗ GPD 拟合失败: {str(e)}")
            # Fallback: 用经验分位数
            self.gpd_params = None

    def calculate_var(self):
        """
        计算 EVT-VaR（基于 GPD 的 Value at Risk）

        公式（POT 方法）:
        VaR_α = u + (σ/ξ) * [ ((1-α)/(1-u_percentile))^(-ξ) - 1 ]

        其中:
        - u: 阈值
        - σ, ξ: GPD 参数
        - α: 置信水平（如 0.99）
        """
        if self.gpd_params is None:
            # Fallback: 使用经验分位数
            self.var = self.returns.quantile(self.confidence)
            print(f"\n使用经验分位数: {self.confidence*100}% VaR = {self.var:.2f} bps")
            return self.var

        shape = self.gpd_params['shape']
        scale = self.gpd_params['scale']

        # 超过阈值的概率
        p_exceed = 1 - self.threshold_percentile

        # EVT-VaR 公式
        if abs(shape) < 1e-6:  # shape ≈ 0 时，用指数分布公式
            self.var = self.threshold + scale * np.log((1 - self.confidence) / p_exceed)
        else:
            self.var = self.threshold + (scale / shape) * (
                ((1 - self.confidence) / p_exceed) ** (-shape) - 1
            )

        print(f"\n" + "="*60)
        print(f"🎯 EVT-VaR ({self.confidence*100}% 置信水平)")
        print(f"   最大日损失预期: {self.var:.2f} bps")
        print(f"   解读: 在 100 个交易日中，最坏的那一天利差扩大不超过此值")
        print("="*60)

        return self.var

    def get_tail_index(self):
        """返回尾部指数（Heavy-tail Index）"""
        if self.gpd_params is None:
            return None
        # 尾部指数 = 1/ξ（ξ 越大，尾部越重）
        return 1 / self.gpd_params['shape'] if self.gpd_params['shape'] > 0 else np.inf
