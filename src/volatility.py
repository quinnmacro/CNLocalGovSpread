"""
波动率建模模块 - 实现 GARCH 锦标赛

技术细节:
- 使用 Student-t 分布而非正态分布（地方债收益率有肥尾）
- 设置较宽松的收敛容差（tol=1e-4），避免优化器在极端行情下卡住
- 自动保存每个模型的 AIC/BIC，最后选出 Winner
"""

import numpy as np
import pandas as pd
from arch import arch_model


class VolatilityModeler:
    """
    波动率建模类 - 实现 GARCH 锦标赛

    技术细节:
    - 使用 Student-t 分布而非正态分布（地方债收益率有肥尾）
    - 设置较宽松的收敛容差（tol=1e-4），避免优化器在极端行情下卡住
    - 自动保存每个模型的 AIC/BIC，最后选出 Winner
    """

    def __init__(self, returns, p=1, q=1):
        self.returns = returns
        self.p = p
        self.q = q
        self.models = {}
        self.results = {}
        self.ic_scores = {}

    def fit_garch(self):
        """标准 GARCH(1,1) - 对称模型基准"""
        print("\n[1/3] 拟合 GARCH(1,1)...")
        try:
            # vol='GARCH' 是默认选项，这里显式写出来是为了代码可读性
            model = arch_model(self.returns, vol='Garch', p=self.p, q=self.q, dist='t')

            # 这个 tol 很关键：默认的 1e-6 在中国市场的某些极端时期会导致不收敛
            # 我们设成 1e-4 牺牲一点精度换取稳定性
            result = model.fit(disp='off', options={'ftol': 1e-4, 'maxiter': 500})

            self.models['GARCH'] = model
            self.results['GARCH'] = result
            self.ic_scores['GARCH'] = {'AIC': result.aic, 'BIC': result.bic}
            print(f"   AIC={result.aic:.2f}, BIC={result.bic:.2f}")

        except Exception as e:
            print(f"   ✗ GARCH 拟合失败: {str(e)}")
            self.ic_scores['GARCH'] = {'AIC': np.inf, 'BIC': np.inf}

    def fit_egarch(self):
        """EGARCH - 对数波动率模型（Nelson 1991）"""
        print("\n[2/3] 拟合 EGARCH(1,1)...")
        try:
            model = arch_model(self.returns, vol='EGARCH', p=self.p, q=self.q, dist='t')
            result = model.fit(disp='off', options={'ftol': 1e-4, 'maxiter': 500})

            self.models['EGARCH'] = model
            self.results['EGARCH'] = result
            self.ic_scores['EGARCH'] = {'AIC': result.aic, 'BIC': result.bic}
            print(f"   AIC={result.aic:.2f}, BIC={result.bic:.2f}")

            # EGARCH 的关键参数：gamma（不对称系数）
            # gamma < 0 意味着负冲击（利差扩大）会放大波动率
            if 'gamma[1]' in result.params:
                gamma = result.params['gamma[1]']
                print(f"   非对称系数 γ = {gamma:.4f} {'(负冲击放大波动)' if gamma < 0 else '(正冲击放大波动)'}")

        except Exception as e:
            print(f"   ✗ EGARCH 拟合失败: {str(e)}")
            self.ic_scores['EGARCH'] = {'AIC': np.inf, 'BIC': np.inf}

    def fit_gjr_garch(self):
        """GJR-GARCH - 阈值模型（Glosten, Jagannathan, Runkle 1993）"""
        print("\n[3/3] 拟合 GJR-GARCH(1,1)...")
        try:
            # vol='GARCH' + o=1 就是 GJR-GARCH
            # o 代表 asymmetric term (threshold effect)
            model = arch_model(self.returns, vol='Garch', p=self.p, o=1, q=self.q, dist='t')
            result = model.fit(disp='off', options={'ftol': 1e-4, 'maxiter': 500})

            self.models['GJR-GARCH'] = model
            self.results['GJR-GARCH'] = result
            self.ic_scores['GJR-GARCH'] = {'AIC': result.aic, 'BIC': result.bic}
            print(f"   AIC={result.aic:.2f}, BIC={result.bic:.2f}")

        except Exception as e:
            print(f"   ✗ GJR-GARCH 拟合失败: {str(e)}")
            self.ic_scores['GJR-GARCH'] = {'AIC': np.inf, 'BIC': np.inf}

    def fit_ewma(self, lambda_param=0.94):
        """
        EWMA (Exponentially Weighted Moving Average) 波动率模型

        RiskMetrics 标准方法:
        - lambda = 0.94 (日频数据标准值)
        - 公式: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}

        优点: 无需优化，计算简单，业界标准基准

        参数:
        - lambda_param: 衰减因子，默认0.94 (RiskMetrics标准)
        """
        print("\n[4/4] 计算 EWMA 波动率...")

        returns = self.returns.values
        n = len(returns)

        # 初始化方差序列
        variance = np.zeros(n)
        variance[0] = returns[0] ** 2

        # EWMA 递推
        for t in range(1, n):
            variance[t] = lambda_param * variance[t-1] + (1 - lambda_param) * returns[t-1] ** 2

        volatility = np.sqrt(variance)

        # 创建 pandas Series
        ewma_volatility = pd.Series(volatility, index=self.returns.index)

        # 计算对数似然 (假设正态分布)
        # 避免 log(0) 的问题
        valid_idx = variance[1:] > 0
        log_likelihood = -0.5 * np.sum(
            np.log(variance[1:][valid_idx]) + returns[1:][valid_idx]**2 / variance[1:][valid_idx]
        )

        # AIC: 只有一个参数 lambda
        aic = 2 * 1 - 2 * log_likelihood
        bic = aic  # 单参数模型

        self.models['EWMA'] = {'volatility': ewma_volatility, 'lambda': lambda_param}
        self.ic_scores['EWMA'] = {'AIC': aic, 'BIC': bic}

        print(f"   λ = {lambda_param} (RiskMetrics 标准)")
        print(f"   AIC={aic:.2f}, BIC={bic:.2f}")

        return ewma_volatility

    def run_tournament(self):
        """执行锦标赛 - 拟合所有模型并选出 Winner"""
        print("\n" + "="*60)
        print("开始波动率模型锦标赛")
        print("="*60)

        self.fit_garch()
        self.fit_egarch()
        self.fit_gjr_garch()
        self.fit_ewma()  # 新增 EWMA 基准模型

        # 根据 AIC 选出最佳模型（AIC 越小越好）
        # 为什么用 AIC 而不是 BIC？BIC 对参数数量惩罚更重，可能过于保守
        winner = min(self.ic_scores, key=lambda x: self.ic_scores[x]['AIC'])

        print("\n" + "="*60)
        print(f"🏆 锦标赛获胜者: {winner}")
        print(f"   AIC = {self.ic_scores[winner]['AIC']:.2f}")
        print(f"   BIC = {self.ic_scores[winner]['BIC']:.2f}")
        print("="*60)

        return winner

    def get_conditional_volatility(self, model_name):
        """提取条件波动率序列"""
        if model_name == 'EWMA':
            # EWMA 存储在 models 字典中
            if 'EWMA' not in self.models:
                raise ValueError("EWMA 模型未计算")
            return self.models['EWMA']['volatility']
        else:
            if model_name not in self.results:
                raise ValueError(f"模型 {model_name} 未拟合")
            # conditional_volatility 是 GARCH 模型的核心输出
            # 它告诉我们「在给定历史信息下，今天的预期波动率是多少」
            return self.results[model_name].conditional_volatility
