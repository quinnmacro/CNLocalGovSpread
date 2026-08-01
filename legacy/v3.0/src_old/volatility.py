"""
波动率建模模块 - 实现 GARCH 锦标赛

技术细节:
- 使用 Student-t 分布而非正态分布（地方债收益率有肥尾）
- 设置较宽松的收敛容差（tol=1e-4），避免优化器在极端行情下卡住
- 自动保存每个模型的 AIC/BIC，最后选出 Winner
- FIGARCH 长记忆检测（v3.0新增）: GPH估计器检测波动率长记忆特性
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats, optimize


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

            # P0修复: 检查收敛状态（arch库使用convergence_flag，0=成功）
            converged = result.convergence_flag == 0
            if not converged:
                print(f"   ⚠️ GARCH 优化未收敛（可能影响参数准确性）")

            self.models['GARCH'] = model
            self.results['GARCH'] = result
            self.ic_scores['GARCH'] = {'AIC': result.aic, 'BIC': result.bic, 'converged': converged}
            print(f"   AIC={result.aic:.2f}, BIC={result.bic:.2f}, 收敛={converged}")

        except Exception as e:
            print(f"   ✗ GARCH 拟合失败: {str(e)}")
            self.ic_scores['GARCH'] = {'AIC': np.inf, 'BIC': np.inf}

    def fit_egarch(self):
        """EGARCH - 对数波动率模型（Nelson 1991）"""
        print("\n[2/3] 拟合 EGARCH(1,1)...")
        try:
            model = arch_model(self.returns, vol='EGARCH', p=self.p, q=self.q, dist='t')
            result = model.fit(disp='off', options={'ftol': 1e-4, 'maxiter': 500})

            # P0修复: 检查收敛状态（arch库使用convergence_flag，0=成功）
            converged = result.convergence_flag == 0
            if not converged:
                print(f"   ⚠️ EGARCH 优化未收敛（可能影响参数准确性）")

            self.models['EGARCH'] = model
            self.results['EGARCH'] = result
            self.ic_scores['EGARCH'] = {'AIC': result.aic, 'BIC': result.bic, 'converged': converged}
            print(f"   AIC={result.aic:.2f}, BIC={result.bic:.2f}, 收敛={converged}")

            # P0修复: arch库的EGARCH实现是对称模型，不包含非对称gamma参数
            # 若需要非对称效应分析，请使用GJR-GARCH模型（fit_gjr_garch方法）
            # 注: 标准EGARCH的非对称性通过|z_t|项隐式体现，而非显式gamma参数

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

            # P0修复: 检查收敛状态（arch库使用convergence_flag，0=成功）
            converged = result.convergence_flag == 0
            if not converged:
                print(f"   ⚠️ GJR-GARCH 优化未收敛（可能影响参数准确性）")

            self.models['GJR-GARCH'] = model
            self.results['GJR-GARCH'] = result
            self.ic_scores['GJR-GARCH'] = {'AIC': result.aic, 'BIC': result.bic, 'converged': converged}
            print(f"   AIC={result.aic:.2f}, BIC={result.bic:.2f}, 收敛={converged}")

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

        # P0修复: 检查数据长度，避免空数组索引错误
        if n < 5:
            # 样本太少，使用第一个收益率的平方作为初始方差
            variance[0] = returns[0] ** 2 if n > 0 else 1e-4
        else:
            # 使用前5个收益率的方差作为初始值，更稳定
            variance[0] = np.var(returns[:5])

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

        # P0修复: 使用t分布计算对数似然，与GARCH模型保持一致
        # GARCH模型使用t分布，EWMA若用正态分布则AIC不可直接比较
        from scipy import stats

        valid_idx = variance[1:] > min_variance
        valid_count = np.sum(valid_idx)

        if valid_count < 10:
            # 样本太少，使用简化方法
            log_likelihood = -1e6  # 惩罚值
            aic = 2 * 2 - 2 * log_likelihood  # 2参数: lambda + df
            bic = np.log(valid_count) * 2 - 2 * log_likelihood
        else:
            # 标准化残差
            std_resid = returns[1:][valid_idx] / volatility[1:][valid_idx]

            # 估计t分布自由度（使用标准化残差）
            try:
                df, loc, scale = stats.t.fit(std_resid)
                df = max(2.1, min(df, 30))  # 限制df在合理范围
            except Exception:
                df = 5.0  # 默认值

            # 使用t分布计算对数似然
            log_likelihood = np.sum(stats.t.logpdf(std_resid, df))

            # EWMA有2个参数: lambda 和 t分布自由度df
            aic = 2 * 2 - 2 * log_likelihood
            bic = np.log(valid_count) * 2 - 2 * log_likelihood

        self.models['EWMA'] = {'volatility': ewma_volatility, 'lambda': lambda_param, 'df': df if valid_count >= 10 else 5.0}
        self.ic_scores['EWMA'] = {'AIC': aic, 'BIC': bic, 'converged': True}  # EWMA无需优化迭代

        print(f"   λ = {lambda_param} (RiskMetrics 标准)")
        if valid_count >= 10:
            print(f"   t分布自由度 df = {df:.2f} (P0修复: 与GARCH使用相同分布)")
        print(f"   AIC={aic:.2f}, BIC={bic:.2f}")
        print(f"   当前波动率: {volatility[-1]:.4f}")

        return ewma_volatility

    def run_tournament(self, include_figarch=False, ewma_lambda=None):
        """
        执行锦标赛 - 拟合所有模型并选出 Winner

        参数:
        - include_figarch: 是否包含FIGARCH模型 (默认False, 因计算较慢)
        - ewma_lambda: EWMA衰减因子 (None=使用默认0.94, 否则使用校准值)
        """
        print("\n" + "="*60)
        print("开始波动率模型锦标赛")
        print("="*60)

        self.fit_garch()
        self.fit_egarch()
        self.fit_gjr_garch()
        self.fit_ewma(lambda_param=ewma_lambda or 0.94)
        if include_figarch:
            self.fit_figarch()

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
        if model_name in ('EWMA', 'FIGARCH'):
            # EWMA 和 FIGARCH 存储在 models 字典中 (非arch库结果)
            if model_name not in self.models:
                raise ValueError(f"{model_name} 模型未计算")
            return self.models[model_name]['volatility']
        else:
            if model_name not in self.results:
                raise ValueError(f"模型 {model_name} 未拟合")
            # conditional_volatility 是 GARCH 模型的核心输出
            # 它告诉我们「在给定历史信息下，今天的预期波动率是多少」
            return self.results[model_name].conditional_volatility

    def get_parameter_diagnostics(self, model_name):
        """
        P0修复: 获取模型参数诊断信息（t统计量、p值、标准误差）

        参数显著性检验是验证模型质量的关键步骤：
        - t统计量 > 2 表示参数在95%水平显著
        - p值 < 0.05 表示参数统计显著
        """
        if model_name not in self.results:
            return None

        result = self.results[model_name]

        # arch库的结果对象提供参数汇总表
        try:
            result.summary()  # verify summary available
            # 提取参数名、估计值、标准误差、t统计量、p值
            params = result.params
            pvalues = result.pvalues
            std_errors = result.std_err

            diagnostics = {}
            for param_name in params.index:
                t_stat = params[param_name] / std_errors[param_name] if std_errors[param_name] > 0 else np.inf
                diagnostics[param_name] = {
                    'estimate': params[param_name],
                    'std_error': std_errors[param_name],
                    't_stat': t_stat,
                    'p_value': pvalues[param_name],
                    'significant': pvalues[param_name] < 0.05
                }

            return diagnostics
        except Exception:
            return None

    def detect_long_memory(self, max_freq_frac=0.5):
        """
        GPH 估计器 - 检测波动率长记忆特性 (v3.0新增)

        Geweke-Porter-Hudak (GPH) 半参数估计器：
        - 使用平方收益率的周期图作为波动率代理
        - 在低频段回归 log(I(ω_j)) ~ log(4sin²(ω_j/2))
        - 斜率估计即为分数差分参数 d

        d 的含义:
        - d ≈ 0: 短记忆 (标准GARCH适用)
        - 0 < d < 1: 长记忆 (FIGARCH更合适)
        - d ≈ 1: 无限记忆 (IGARCH)

        参数:
        - max_freq_frac: 使用的最大频率比例 (默认0.5, 即前半段低频)

        返回:
        - dict: {d_estimate, d_std_error, d_t_stat, d_p_value,
                 long_memory_detected, memory_type, gph_regression_details}
        """
        returns = self.returns.values
        n = len(returns)

        if n < 50:
            print("   ⚠️ 数据不足 (<50), 无法进行长记忆检测")
            return {
                'd_estimate': 0.0, 'd_std_error': np.inf,
                'd_t_stat': 0.0, 'd_p_value': 1.0,
                'long_memory_detected': False, 'memory_type': '短记忆 (数据不足)',
                'gph_regression_details': None
            }

        # 使用平方收益率作为波动率代理
        squared_returns = returns ** 2

        # 计算周期图 (离散傅里叶变换功率谱)
        # FFT 输出: [0, ω_1, ..., ω_{n/2}, ω_{-(n/2-1)}, ..., ω_{-1}]
        fft_result = np.fft.fft(squared_returns - np.mean(squared_returns))
        n_freq = n // 2  # 只取正频率

        # 周期图 I(ω_j) = (1/n) |FFT_j|²
        periodogram = np.abs(fft_result[:n_freq + 1]) ** 2 / n

        # 频率索引 j = 1, 2, ..., m (排除 j=0 的零频)
        m = int(max_freq_frac * n_freq)
        if m < 5:
            m = min(n_freq, 5)

        j_indices = np.arange(1, m + 1)
        # 傅里叶频率 ω_j = 2πj/n
        omega_j = 2 * np.pi * j_indices / n

        # 回归变量: X_j = -log(4 * sin²(ω_j/2))
        # 这是分数差分滤波器 (1-L)^d 的谱密度 log项
        x_j = -np.log(4 * np.sin(omega_j / 2) ** 2)

        # 回归因变量: Y_j = log(I(ω_j))
        # 防止 log(0): 添加微小常数
        y_j = np.log(periodogram[j_indices] + 1e-10)

        # OLS 回归: Y = a + d * X
        # d 的估计即为斜率
        X_matrix = np.column_stack([np.ones(m), x_j])
        try:
            beta, residuals_rank, _, _ = np.linalg.lstsq(X_matrix, y_j, rcond=None)
            d_estimate = beta[1]
            intercept = beta[0]

            # 计算残差和标准误差
            y_hat = X_matrix @ beta
            residuals = y_j - y_hat
            rss = np.sum(residuals ** 2)
            # t 分布近似下的标准误差 (Robinson 1995)
            # σ²_d ≈ π²/6m (GPH方差的理论值)
            d_std_error = np.sqrt(np.pi ** 2 / (6 * m))
            d_t_stat = d_estimate / d_std_error

            # 双侧检验 p值
            d_p_value = 2 * stats.norm.sf(np.abs(d_t_stat))

            # 判断记忆类型
            if d_p_value < 0.05 and d_estimate > 0:
                if d_estimate > 0.9:
                    memory_type = '无限记忆 (IGARCH域)'
                elif d_estimate > 0.4:
                    memory_type = '强长记忆'
                else:
                    memory_type = '弱长记忆'
                long_memory = True
            else:
                memory_type = '短记忆 (GARCH适用)'
                long_memory = False

            gph_details = {
                'n_freq_used': m,
                'intercept': intercept,
                'r_squared': 1 - rss / np.sum((y_j - np.mean(y_j)) ** 2),
                'd_confidence_interval': (d_estimate - 1.96 * d_std_error,
                                          d_estimate + 1.96 * d_std_error)
            }

        except Exception as e:
            print(f"   ⚠️ GPH 回归失败: {str(e)}")
            return {
                'd_estimate': 0.0, 'd_std_error': np.inf,
                'd_t_stat': 0.0, 'd_p_value': 1.0,
                'long_memory_detected': False,
                'memory_type': '短记忆 (估计失败)',
                'gph_regression_details': None
            }

        result = {
            'd_estimate': d_estimate,
            'd_std_error': d_std_error,
            'd_t_stat': d_t_stat,
            'd_p_value': d_p_value,
            'long_memory_detected': long_memory,
            'memory_type': memory_type,
            'gph_regression_details': gph_details
        }

        print(f"\n长记忆检测结果 (GPH估计器):")
        print(f"   分数差分参数 d = {d_estimate:.4f} (标准误差: {d_std_error:.4f})")
        print(f"   t统计量 = {d_t_stat:.2f}, p值 = {d_p_value:.4f}")
        print(f"   95%置信区间: [{gph_details['d_confidence_interval'][0]:.4f}, "
              f"{gph_details['d_confidence_interval'][1]:.4f}]")
        print(f"   记忆类型: {memory_type}")
        print(f"   长记忆显著: {long_memory}")

        return result

    def fit_figarch(self, truncation_lag=100, max_freq_frac=0.5):
        """
        FIGARCH(1,d,1) - 分数整合GARCH模型 (v3.0新增)

        FIGARCH 捕捉波动率长记忆:
        - 标准GARCH: 冲击以指数速率衰减 (短记忆)
        - FIGARCH: 冲击以双曲速率衰减 (长记忆, Baillie, Bollerslev, Mikkelsen 1996)
        - 关键参数 d: 分数差分阶数, 0 < d < 1

        实现方法: 截断多项式展开
        (1-L)^d ≈ Σ_{k=0}^{T} π_k L^k
        π_0 = 1, π_k = (k-1-d)/k * π_{k-1}

        参数:
        - truncation_lag: 截断阶数T (默认100)
        - max_freq_frac: GPH估计d时的最大频率比例

        返回:
        - dict: FIGARCH拟合结果 (参数, 条件波动率, IC分数)
        """
        print(f"\n[FIGARCH] 拟合 FIGARCH(1,d,1)...")

        # Step 1: 用GPH估计器检测d
        gph_result = self.detect_long_memory(max_freq_frac=max_freq_frac)
        d_estimate = gph_result['d_estimate']

        # 将d限制在合理范围 [0.01, 0.99]
        d_estimate = max(0.01, min(0.99, d_estimate))

        returns = self.returns.values
        n = len(returns)

        if n < 30:
            print("   ⚠️ 数据不足 (<30), FIGARCH拟合不可靠")
            self.ic_scores['FIGARCH'] = {'AIC': np.inf, 'BIC': np.inf, 'converged': False}
            self.models['FIGARCH'] = None
            return None

        # Step 2: 计算分数差分滤波器系数 π_k
        T = min(truncation_lag, n - 1)
        pi = np.zeros(T + 1)
        pi[0] = 1.0
        for k in range(1, T + 1):
            pi[k] = (k - 1 - d_estimate) / k * pi[k - 1]

        # Step 3: 估计FIGARCH参数 (ω, α, β) via 数值优化
        # FIGARCH(1,d,1) 方差方程:
        # σ²_t = ω + [1 - (1-L)^d] ε²_t + β σ²_{t-1}
        # 展开: σ²_t = ω + Σ_{k=1}^{T} π_k ε²_{t-k} + β σ²_{t-1}
        # 等价于: σ²_t = ω + α ε²_{t-1} + β σ²_{t-1} + Σ_{k=2}^{T} π_k ε²_{t-k}

        # 初始值: 使用GARCH(1,1)参数作为初始猜测
        try:
            garch_model = arch_model(self.returns, vol='Garch', p=1, q=1, dist='t')
            garch_result = garch_model.fit(disp='off', options={'ftol': 1e-4, 'maxiter': 200})
            omega_init = garch_result.params.get('omega', np.var(returns) * 0.1)
            alpha_init = garch_result.params.get('alpha[1]', 0.1)
            beta_init = garch_result.params.get('beta[1]', 0.85)
        except Exception:
            omega_init = np.var(returns) * 0.1
            alpha_init = 0.1
            beta_init = 0.85

        # 似然函数: 计算FIGARCH条件方差和t分布对数似然
        def figarch_loglikelihood(params):
            omega, alpha, beta = params
            # 约束: ω>0, α>0, β≥0, α+β<1 (平稳性)
            if omega <= 0 or alpha <= 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            # 计算条件方差序列
            variance = np.zeros(n)
            variance[0] = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else omega
            eps_sq = returns ** 2

            for t in range(1, n):
                # FIGARCH方差: ω + α * ε²_{t-1} + β * σ²_{t-1} + Σ_{k=2}^{T} π_k * ε²_{t-k}
                long_mem_term = 0.0
                for k in range(2, min(T + 1, t + 1)):
                    long_mem_term += pi[k] * eps_sq[t - k]

                variance[t] = omega + alpha * eps_sq[t - 1] + beta * variance[t - 1] + long_mem_term
                variance[t] = max(variance[t], 1e-10)  # 防止负方差

            # 标准化残差
            vol = np.sqrt(variance)
            std_resid = returns / vol

            # t分布对数似然 (与GARCH锦标赛一致)
            try:
                df, _, _ = stats.t.fit(std_resid[5:])  # 跳过初始不稳定期
                df = max(2.1, min(df, 30))
            except Exception:
                df = 5.0

            ll = np.sum(stats.t.logpdf(std_resid[5:], df))
            return -ll  # 最小化负对数似然

        # Step 4: 数值优化
        try:
            opt_result = optimize.minimize(
                figarch_loglikelihood,
                x0=[omega_init, alpha_init, beta_init],
                method='L-BFGS-B',
                bounds=[(1e-6, None), (1e-6, None), (0, 0.999)],
                options={'maxiter': 300, 'ftol': 1e-4}
            )
            omega, alpha, beta = opt_result.x
            converged = opt_result.success

            # 重新计算最终条件方差
            variance = np.zeros(n)
            variance[0] = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else omega
            eps_sq = returns ** 2

            for t in range(1, n):
                long_mem_term = 0.0
                for k in range(2, min(T + 1, t + 1)):
                    long_mem_term += pi[k] * eps_sq[t - k]
                variance[t] = omega + alpha * eps_sq[t - 1] + beta * variance[t - 1] + long_mem_term
                variance[t] = max(variance[t], 1e-10)

            vol = np.sqrt(variance)
            figarch_volatility = pd.Series(vol, index=self.returns.index)

            # 计算IC分数
            std_resid = returns / vol
            try:
                df, _, _ = stats.t.fit(std_resid[5:])
                df = max(2.1, min(df, 30))
            except Exception:
                df = 5.0

            ll = np.sum(stats.t.logpdf(std_resid[5:], df))
            # 参数数量: ω, α, β, d, df = 5
            k_params = 5
            n_obs = n - 5
            aic = 2 * k_params - 2 * ll
            bic = np.log(n_obs) * k_params - 2 * ll

            figarch_result = {
                'volatility': figarch_volatility,
                'params': {'omega': omega, 'alpha': alpha, 'beta': beta, 'd': d_estimate, 'df': df},
                'pi_coefficients': pi[:min(T + 1, 20)],  # 存储前20个系数供诊断
                'gph_result': gph_result,
                'converged': converged
            }

            self.models['FIGARCH'] = figarch_result
            self.ic_scores['FIGARCH'] = {'AIC': aic, 'BIC': bic, 'converged': converged}

            print(f"   参数: ω={omega:.6f}, α={alpha:.4f}, β={beta:.4f}, d={d_estimate:.4f}")
            print(f"   AIC={aic:.2f}, BIC={bic:.2f}, 收敛={converged}")
            print(f"   当前波动率: {vol[-1]:.4f}")
            print(f"   长记忆系数π衰减: π_5={pi[5]:.4f}, π_20={pi[20]:.4f} "
                  f"(双曲衰减 vs GARCH指数衰减)")

            return figarch_result

        except Exception as e:
            print(f"   ✗ FIGARCH 拟合失败: {str(e)}")
            self.ic_scores['FIGARCH'] = {'AIC': np.inf, 'BIC': np.inf, 'converged': False}
            self.models['FIGARCH'] = None
            return None


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
        try:
            from hmmlearn import hmm
        except ImportError:
            print("⚠️ hmmlearn 未安装，跳过状态检测")
            self.regime_labels = np.zeros(len(self.volatility), dtype=int)
            self.regime_stats = {0: {'mean': self.volatility.mean(), 'std': self.volatility.std(), 'count': len(self.volatility), 'pct': 100.0}}
            return self.regime_labels

        # 准备数据
        X = self.volatility.values.reshape(-1, 1)

        # P0修复: 添加异常处理
        try:
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
        except Exception as e:
            print(f"⚠️ HMM拟合失败: {str(e)}，使用默认状态")
            self.regime_labels = np.zeros(len(self.volatility), dtype=int)

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
