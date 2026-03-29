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
        # 使用前5个收益率的方差作为初始值，更稳定
        variance[0] = np.var(returns[:min(5, n)]) if n >= 5 else returns[0] ** 2

        # 添加方差下限，防止数值下溢
        min_variance = 1e-10

        # EWMA 递推
        for t in range(1, n):
            variance[t] = lambda_param * variance[t-1] + (1 - lambda_param) * returns[t-1] ** 2
            # 防止方差过小
            variance[t] = max(variance[t], min_variance)

        volatility = np.sqrt(variance)

        # 创建 pandas Series
        ewma_volatility = pd.Series(volatility, index=self.returns.index)

        # 计算对数似然 (假设正态分布)
        # 避免 log(0) 和数值溢出
        valid_idx = variance[1:] > min_variance
        valid_count = np.sum(valid_idx)

        if valid_count < 10:
            # 样本太少，使用简化方法
            log_likelihood = -1e6  # 惩罚值
            aic = 2 * 1 - 2 * log_likelihood
            bic = aic
        else:
            # 安全计算对数似然
            log_terms = np.log(variance[1:][valid_idx])
            ratio_terms = returns[1:][valid_idx]**2 / variance[1:][valid_idx]

            # 检查是否有异常值
            if np.any(np.isnan(log_terms)) or np.any(np.isinf(log_terms)):
                log_likelihood = -1e6
            elif np.any(np.isnan(ratio_terms)) or np.any(np.isinf(ratio_terms)):
                log_likelihood = -1e6
            else:
                log_likelihood = -0.5 * np.sum(log_terms + ratio_terms)

            aic = 2 * 1 - 2 * log_likelihood
            bic = aic  # 单参数模型

        self.models['EWMA'] = {'volatility': ewma_volatility, 'lambda': lambda_param}
        self.ic_scores['EWMA'] = {'AIC': aic, 'BIC': bic}

        print(f"   λ = {lambda_param} (RiskMetrics 标准)")
        print(f"   AIC={aic:.2f}, BIC={bic:.2f}")
        print(f"   当前波动率: {volatility[-1]:.4f}")

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


class RegimeDetector:
    """
    波动率状态切换检测器

    使用隐马尔可夫模型(HMM)识别不同的波动率状态：
    - 状态0: 低波动
    - 状态1: 中波动
    - 状态2: 高波动
    """

    def __init__(self, volatility_series, n_regimes=3):
        """
        参数:
        - volatility_series: 条件波动率序列
        - n_regimes: 状态数量，默认3个 (低/中/高)
        """
        self.volatility = volatility_series
        self.n_regimes = n_regimes
        self.model = None
        self.regime_labels = None
        self.regime_stats = None

    def fit(self):
        """
        拟合HMM模型

        输出:
        - regime_labels: 每个时点的状态标签 (0, 1, 2)
        - regime_stats: 每个状态的统计特征
        """
        from hmmlearn import hmm

        # 准备数据
        X = self.volatility.values.reshape(-1, 1)

        # 拟合高斯HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        self.model.fit(X)

        # 预测状态
        self.regime_labels = self.model.predict(X)

        # 计算每个状态的统计特征
        self.regime_stats = {}
        for i in range(self.n_regimes):
            mask = self.regime_labels == i
            vol_in_regime = self.volatility[mask]
            self.regime_stats[i] = {
                'mean': vol_in_regime.mean(),
                'std': vol_in_regime.std(),
                'count': len(vol_in_regime),
                'pct': len(vol_in_regime) / len(self.volatility) * 100
            }

        # 按均值排序状态 (0=低, 1=中, 2=高)
        means = [self.regime_stats[i]['mean'] for i in range(self.n_regimes)]
        sorted_order = np.argsort(means)
        label_mapping = {old: new for new, old in enumerate(sorted_order)}
        self.regime_labels = np.array([label_mapping[l] for l in self.regime_labels])

        # 更新统计特征
        new_stats = {}
        for old_label, new_label in label_mapping.items():
            new_stats[new_label] = self.regime_stats[old_label]
        self.regime_stats = new_stats

        return self.regime_labels

    def get_current_regime(self):
        """返回当前状态"""
        if self.regime_labels is None:
            raise ValueError("请先调用 fit()")
        return self.regime_labels[-1]

    def get_regime_name(self, regime_id):
        """返回状态名称"""
        names = {0: '低波动', 1: '中波动', 2: '高波动'}
        return names.get(regime_id, f'状态{regime_id}')

    def print_regime_summary(self):
        """打印状态摘要"""
        print("\n" + "="*60)
        print("波动率状态切换分析")
        print("="*60)

        for i in range(self.n_regimes):
            stats = self.regime_stats[i]
            print(f"\n{self.get_regime_name(i)}:")
            print(f"  平均波动率: {stats['mean']:.2f} bps")
            print(f"  波动率标准差: {stats['std']:.2f} bps")
            print(f"  样本占比: {stats['pct']:.1f}%")

        current = self.get_current_regime()
        print(f"\n当前状态: {self.get_regime_name(current)}")
        print("="*60)
