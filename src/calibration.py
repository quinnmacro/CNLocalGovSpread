"""
参数自动校准模块 - 从真实数据估计模型参数，替代硬编码默认值

核心功能:
1. EWMA lambda优化 - 最大化方差预测精度
2. t分布df估计 - 从残差MLE拟合
3. AR(1) phi估计 - OLS回归
4. EVT阈值优化 - Mean Excess Plot稳定性分析
5. Kalman窗口优化 - 交叉验证平滑误差
6. GARCH持久性校验 - alpha+beta<1约束诊断
7. 信号偏离阈值估计 - 数据驱动分位数

设计理念:
- 所有估计方法均有理论基础和文献引用
- 每个估计附带置信区间和诊断信息
- 自动检测数据质量并调整估计策略
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize


class ParameterCalibrator:
    """
    参数自动校准器 - 从数据估计lambda/phi/df等模型参数

    替代的硬编码默认值:
    - EWMA lambda = 0.94 (RiskMetrics标准)
    - t分布 df = 5.0 (fallback)
    - AR(1) phi = 0.98 (mock数据)
    - EVT threshold = 0.95 (McNeil & Frey经验法则)
    - Kalman窗口 = 60 (季度经验)
    - 信号偏离阈值 = 1.5 sigma (交易经验)
    """

    def __init__(self, returns, spread=None):
        """
        参数:
        - returns: pd.Series, 利差变化序列
        - spread: pd.Series, 原始利差序列（可选，用于Kalman窗口优化）
        """
        self.returns = returns
        self.spread = spread
        self.calibrated = {}
        self.diagnostics = {}

    def calibrate_all(self):
        """一键校准所有参数"""
        print("\n" + "="*60)
        print("参数自动校准 (Parameter Auto-Calibration)")
        print("="*60)

        self.estimate_ewma_lambda()
        self.estimate_t_df()
        self.estimate_ar_phi()
        self.optimize_evt_threshold()
        self.optimize_kalman_window()
        self.estimate_signal_threshold()

        print("\n" + "="*60)
        print("校准结果汇总")
        print("="*60)
        for param, value in self.calibrated.items():
            diag = self.diagnostics.get(param, {})
            method = diag.get('method', 'N/A')
            ci = diag.get('ci', 'N/A')
            print(f"  {param}: {value:.4f} (方法: {method}, 置信区间: {ci})")
        print("="*60)

        return self.calibrated

    def estimate_ewma_lambda(self, lambda_range=(0.80, 0.99)):
        """
        从数据优化EWMA衰减因子lambda

        方法: 最小化方差预测RMSE（Engle 2001, MFE第4章）
        EWMA递推: sigma2_t = lambda * sigma2_{t-1} + (1-lambda) * r2_{t-1}

        评估标准:
        - RMSE: 预测方差 vs 实现收益率平方
        - QLIKE: 对数损失函数（Patton 2011推荐）
        - MAD: 绝对偏差

        参数:
        - lambda_range: 搜索范围，默认(0.80, 0.99)

        返回:
        - float: 最优lambda值
        """
        print("\n[1/6] 优化EWMA衰减因子 λ...")
        returns = self.returns.values
        n = len(returns)

        if n < 30:
            print("   ⚠️ 数据不足30个观测值，使用RiskMetrics默认值 λ=0.94")
            self.calibrated['ewma_lambda'] = 0.94
            self.diagnostics['ewma_lambda'] = {'method': 'fallback', 'ci': 'N/A', 'n': n}
            return 0.94

        def _ewma_loss(lam):
            """计算EWMA预测误差（QLIKE损失，Patton 2011）"""
            variance = np.zeros(n)
            variance[0] = np.var(returns[:5]) if n >= 5 else returns[0] ** 2
            for t in range(1, n):
                variance[t] = lam * variance[t-1] + (1 - lam) * returns[t-1] ** 2
                variance[t] = max(variance[t], 1e-10)

            # QLIKE损失: L = (r2/h) - log(r2/h) - 1
            # 最小化QLIKE ≈ 最小化方差预测误差
            realized = returns[1:] ** 2
            predicted = variance[1:]
            valid = predicted > 1e-10
            if np.sum(valid) < 5:
                return 1e6
            ratio = realized[valid] / predicted[valid]
            # 防止log(0): ratio=0时QLIKE项=ratio-(-inf)-1→-inf，跳过
            safe_ratio = ratio[ratio > 1e-10]
            if len(safe_ratio) < 5:
                return 1e6
            qlike = np.mean(safe_ratio - np.log(safe_ratio) - 1)
            return qlike

        # 网格搜索 + 精细优化
        grid_lambdas = np.linspace(lambda_range[0], lambda_range[1], 50)
        grid_losses = [_ewma_loss(lam) for lam in grid_lambdas]
        best_grid_idx = np.argmin(grid_losses)
        best_lambda_grid = grid_lambdas[best_grid_idx]

        try:
            result = optimize.minimize_scalar(
                _ewma_loss,
                bounds=(max(lambda_range[0], best_lambda_grid - 0.02),
                        min(lambda_range[1], best_lambda_grid + 0.02)),
                method='bounded'
            )
            best_lambda = result.x
        except Exception:
            best_lambda = best_lambda_grid

        # 计算RMSE作为辅助诊断
        variance_opt = np.zeros(n)
        variance_opt[0] = np.var(returns[:5]) if n >= 5 else returns[0] ** 2
        for t in range(1, n):
            variance_opt[t] = best_lambda * variance_opt[t-1] + (1 - best_lambda) * returns[t-1] ** 2
            variance_opt[t] = max(variance_opt[t], 1e-10)

        rmse = np.sqrt(np.mean((returns[1:] ** 2 - variance_opt[1:]) ** 2))

        # 置信区间: 基于网格搜索的loss曲线形状
        # 找到loss不超过最优loss * 1.05的lambda范围
        threshold_loss = grid_losses[best_grid_idx] * 1.05
        valid_mask = grid_losses <= threshold_loss
        ci_low = grid_lambdas[valid_mask].min() if np.any(valid_mask) else best_lambda
        ci_high = grid_lambdas[valid_mask].max() if np.any(valid_mask) else best_lambda

        self.calibrated['ewma_lambda'] = best_lambda
        self.diagnostics['ewma_lambda'] = {
            'method': 'QLIKE优化 (Patton 2011)',
            'ci': f'({ci_low:.4f}, {ci_high:.4f})',
            'rmse': rmse,
            'n': n
        }

        print(f"   λ_opt = {best_lambda:.4f} (vs RiskMetrics默认 0.94)")
        print(f"   RMSE = {rmse:.6f}")
        print(f"   置信区间: ({ci_low:.4f}, {ci_high:.4f})")

        return best_lambda

    def estimate_t_df(self, bounds=(2.1, 30.0)):
        """
        从数据估计Student-t分布自由度df

        方法: 对标准化残差进行MLE拟合（Harvey & Siddique 1999）
        - 使用 scipy.stats.t.fit 估计 df, loc, scale
        - 限制 df 在合理范围 (2.1, 30)

        当 df → ∞ 时退化为正态分布，说明无肥尾效应
        当 df < 5 时表示极端肥尾（需特别关注风险）

        参数:
        - bounds: df搜索范围

        返回:
        - float: 最优df值
        """
        print("\n[2/6] 估计t分布自由度 df...")
        returns = self.returns.values
        n = len(returns)

        if n < 50:
            print("   ⚠️ 数据不足50个观测值，使用默认值 df=5.0")
            self.calibrated['t_df'] = 5.0
            self.diagnostics['t_df'] = {'method': 'fallback', 'ci': 'N/A', 'n': n}
            return 5.0

        # MLE拟合t分布
        try:
            df_est, loc_est, scale_est = stats.t.fit(returns)

            # 限制在合理范围
            df_est = max(bounds[0], min(df_est, bounds[1]))
        except Exception:
            print("   ⚠️ t分布拟合失败，使用默认值 df=5.0")
            self.calibrated['t_df'] = 5.0
            self.diagnostics['t_df'] = {'method': 'MLE_failed', 'ci': 'N/A', 'n': n}
            return 5.0

        # 置信区间: Profile likelihood方法
        # 简化实现：通过不同df值的log-likelihood差来估计
        n_profile = 50
        df_grid = np.linspace(bounds[0], bounds[1], n_profile)
        log_lik = []
        for df_val in df_grid:
            try:
                ll = np.sum(stats.t.logpdf(returns, df_val, loc=loc_est, scale=scale_est))
                log_lik.append(ll)
            except Exception:
                log_lik.append(-1e10)

        log_lik = np.array(log_lik)
        max_ll = log_lik.max()

        # 95% CI: log-likelihood差 < 1.92 (chi2(1)的95%分位数/2)
        ci_mask = log_lik > max_ll - 1.92
        ci_low = df_grid[ci_mask].min() if np.any(ci_mask) else bounds[0]
        ci_high = df_grid[ci_mask].max() if np.any(ci_mask) else bounds[1]

        # 肥尾诊断
        if df_est < 5:
            tail_msg = "极端肥尾 (df<5: 方差可能不稳定)"
        elif df_est < 10:
            tail_msg = "中度肥尾 (df<10: 仍需t分布建模)"
        elif df_est < 25:
            tail_msg = "轻度肥尾 (接近正态)"
        else:
            tail_msg = "近似正态分布 (df>25)"

        self.calibrated['t_df'] = df_est
        self.diagnostics['t_df'] = {
            'method': 'MLE (Harvey & Siddique 1999)',
            'ci': f'({ci_low:.2f}, {ci_high:.2f})',
            'loc': loc_est,
            'scale': scale_est,
            'tail_diagnosis': tail_msg,
            'n': n
        }

        print(f"   df_opt = {df_est:.2f} (vs 硬编码默认 5.0)")
        print(f"   诊断: {tail_msg}")
        print(f"   置信区间: ({ci_low:.2f}, {ci_high:.2f})")
        print(f"   loc={loc_est:.4f}, scale={scale_est:.4f}")

        return df_est

    def estimate_ar_phi(self, bounds=(0.0, 0.99)):
        """
        从数据估计AR(1)系数phi

        方法: OLS回归 (Hamilton 1994, Chapter 3)
        y_t = mu + phi * (y_{t-1} - mu) + epsilon_t
        简化: r_t = phi * r_{t-1} + epsilon_t (对利差变化)

        参数:
        - bounds: phi搜索范围 (0, 0.99)，避免单位根

        返回:
        - float: 最优phi值
        """
        print("\n[3/6] 估计AR(1)系数 φ...")
        returns = self.returns.values
        n = len(returns)

        if n < 20:
            print("   ⚠️ 数据不足20个观测值，使用默认值 φ=0.5")
            self.calibrated['ar_phi'] = 0.5
            self.diagnostics['ar_phi'] = {'method': 'fallback', 'ci': 'N/A', 'n': n}
            return 0.5

        # OLS: r_t = phi * r_{t-1} + epsilon_t
        y = returns[1:]
        x = returns[:-1]

        # numpy OLS
        phi_est = np.sum(x * y) / np.sum(x ** 2) if np.sum(x ** 2) > 0 else 0.0

        # 限制在合理范围
        phi_est = max(bounds[0], min(phi_est, bounds[1]))

        # 计算残差和标准误差
        residuals = y - phi_est * x
        residual_var = np.var(residuals)
        phi_se = np.sqrt(residual_var / np.sum(x ** 2)) if np.sum(x ** 2) > 0 else 0.0

        # t统计量和p值
        t_stat = phi_est / phi_se if phi_se > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        # 95%置信区间
        ci_low = max(bounds[0], phi_est - 1.96 * phi_se)
        ci_high = min(bounds[1], phi_est + 1.96 * phi_se)

        # 持久性诊断
        if phi_est > 0.9:
            persist_msg = "高持久性 (φ>0.9: 利差变化缓慢衰减)"
        elif phi_est > 0.5:
            persist_msg = "中等持久性 (φ>0.5: 部分均值回归)"
        elif phi_est > 0:
            persist_msg = "低持久性 (φ>0: 弱序列相关)"
        else:
            persist_msg = "负相关 (φ<0: 可能均值回归振荡)"

        self.calibrated['ar_phi'] = phi_est
        self.diagnostics['ar_phi'] = {
            'method': 'OLS (Hamilton 1994)',
            'ci': f'({ci_low:.4f}, {ci_high:.4f})',
            'std_error': phi_se,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'persist_diagnosis': persist_msg,
            'n': n
        }

        print(f"   φ_opt = {phi_est:.4f} (vs mock数据默认 0.98)")
        print(f"   标准误差: {phi_se:.4f}")
        print(f"   t统计量: {t_stat:.4f}, p值: {p_value:.4f}")
        print(f"   置信区间: ({ci_low:.4f}, {ci_high:.4f})")
        print(f"   诊断: {persist_msg}")

        return phi_est

    def optimize_evt_threshold(self, percentile_range=(0.85, 0.99)):
        """
        从数据优化EVT阈值百分位数

        方法: Mean Excess Function稳定性分析 (McNeil & Frey 2000)
        - MEF在阈值之上应近似线性（GPD拟合假设）
        - 选择MEF开始稳定的最低百分位数

        自动选择策略:
        1. 计算各百分位下的MEF
        2. 检测MEF线性区域起始点
        3. 平衡样本量 vs 拟合质量

        参数:
        - percentile_range: 搜索范围

        返回:
        - float: 最优阈值百分位数
        """
        print("\n[4/6] 优化EVT阈值百分位数...")
        data = self.returns.values
        n = len(data)

        if n < 100:
            print("   ⚠️ 数据不足100个观测值，使用默认值 0.95")
            self.calibrated['evt_threshold_percentile'] = 0.95
            self.diagnostics['evt_threshold_percentile'] = {'method': 'fallback', 'n': n}
            return 0.95

        # 计算不同百分位下的Mean Excess Function
        percentiles = np.linspace(percentile_range[0], percentile_range[1], 30)
        mef_values = []
        n_exceedances = []

        for pct in percentiles:
            threshold = np.percentile(data, pct * 100)
            exceedances = data[data > threshold] - threshold
            if len(exceedances) > 5:
                mef_values.append(np.mean(exceedances))
                n_exceedances.append(len(exceedances))
            else:
                mef_values.append(np.nan)
                n_exceedances.append(len(exceedances))

        mef_values = np.array(mef_values)
        n_exceedances = np.array(n_exceedances)

        valid_mask = ~np.isnan(mef_values) & (n_exceedances >= 10)
        if np.sum(valid_mask) < 5:
            print("   ⚠️ 有效MEF点不足，使用默认值 0.95")
            self.calibrated['evt_threshold_percentile'] = 0.95
            self.diagnostics['evt_threshold_percentile'] = {'method': 'insufficient_data', 'n': n}
            return 0.95

        valid_pcts = percentiles[valid_mask]
        valid_mef = mef_values[valid_mask]
        valid_n = n_exceedances[valid_mask]

        # 寻找MEF线性区域起始点
        # 方法: 计算MEF变化率，找到变化率稳定的最低百分位数
        diffs = np.diff(valid_mef)
        if len(diffs) < 3:
            optimal_pct = 0.95
        else:
            # 线性度度量: 局部斜率的方差越小越线性
            best_pct_idx = 0
            best_score = np.inf

            for i in range(len(diffs)):
                if i + 5 > len(diffs):
                    break
                # 线性度得分 = 局部斜率方差 + 样本量惩罚
                local_var = np.var(diffs[i:i+5])
                # 样本量惩罚: 超过阈值的样本太少则不可靠
                sample_penalty = max(0, (30 - valid_n[i]) / 30) * 0.5
                score = local_var + sample_penalty
                if score < best_score:
                    best_score = score
                    best_pct_idx = i

            optimal_pct = valid_pcts[best_pct_idx]

        # 计算对应的阈值和样本数
        optimal_threshold = np.percentile(data, optimal_pct * 100)
        optimal_n_exceed = np.sum(data > optimal_threshold)

        # 置信区间: MEF稳定区域对应的百分位范围
        stability_mask = valid_n >= 10
        ci_low_pct = valid_pcts[stability_mask].min() if np.any(stability_mask) else 0.90
        ci_high_pct = valid_pcts[stability_mask].max() if np.any(stability_mask) else 0.99

        self.calibrated['evt_threshold_percentile'] = optimal_pct
        self.diagnostics['evt_threshold_percentile'] = {
            'method': 'MEF稳定性 (McNeil & Frey 2000)',
            'ci': f'({ci_low_pct:.2f}, {ci_high_pct:.2f})',
            'threshold_value': optimal_threshold,
            'n_exceedances': optimal_n_exceed,
            'n': n
        }

        print(f"   pct_opt = {optimal_pct:.2f} (vs 经验法则默认 0.95)")
        print(f"   对应阈值: {optimal_threshold:.2f} bps")
        print(f"   超过阈值的样本: {optimal_n_exceed} 个")

        return optimal_pct

    def optimize_kalman_window(self, window_range=(20, 120)):
        """
        从数据优化Kalman滤波fallback窗口大小

        方法: 交叉验证平滑误差
        - 当Kalman优化失败时使用滚动均值作为fallback
        - 窗口大小影响平滑程度: 太短追踪噪音，太长反应迟钝
        - 优化标准: 最小化平滑值与真实信号的偏离

        参数:
        - window_range: 窗口搜索范围

        返回:
        - int: 最优窗口大小
        """
        print("\n[5/6] 优化Kalman fallback窗口大小...")
        if self.spread is None:
            print("   ⚠️ 未提供原始利差序列，使用默认窗口 60")
            self.calibrated['kalman_window'] = 60
            self.diagnostics['kalman_window'] = {'method': 'no_spread_data'}
            return 60

        spread = self.spread.values
        n = len(spread)

        if n < 60:
            print("   ⚠️ 数据不足60个观测值，使用默认窗口 60")
            self.calibrated['kalman_window'] = 60
            self.diagnostics['kalman_window'] = {'method': 'insufficient_data', 'n': n}
            return 60

        # 交叉验证: 比较不同窗口的平滑误差
        # 使用"差分后重构"作为真实信号代理
        # 即先差分(去趋势)再重构(平滑趋势)
        windows = range(window_range[0], window_range[1] + 1, 5)
        best_window = 60
        best_error = np.inf

        # 真实信号代理: 使用HP滤波趋势（如果可用）或大窗口均值
        # 大窗口(120日)均值作为基准趋势
        reference_trend = pd.Series(spread).rolling(window=min(120, n), min_periods=1).mean().values

        for w in windows:
            smoothed = pd.Series(spread).rolling(window=w, min_periods=1).mean().values
            # 误差: 平滑值偏离基准趋势的程度 + 变化率惩罚
            # 变化率惩罚确保平滑后序列不会过于迟钝
            deviation_error = np.mean((smoothed[n//2:] - reference_trend[n//2:]) ** 2)
            # 变化率: 平滑后序列的一阶差分方差（太低说明反应迟钝）
            change_rate = np.var(np.diff(smoothed[n//2:]))
            # 综合得分: 偏离误差低 + 变化率适中
            score = deviation_error - 0.1 * change_rate  # 略偏好更多变化
            if score < best_error:
                best_error = score
                best_window = w

        self.calibrated['kalman_window'] = best_window
        self.diagnostics['kalman_window'] = {
            'method': '交叉验证 (偏差+变化率平衡)',
            'best_error': best_error,
            'n': n
        }

        print(f"   window_opt = {best_window} (vs 季度经验默认 60)")

        return best_window

    def estimate_signal_threshold(self, confidence=0.95):
        """
        从数据估计信号偏离阈值

        方法: 基于标准化残差的分位数估计
        - 标准化残差 = (观测值 - Kalman平滑值) / σ
        - 阈值 = 标准化残差的分位数 (如95%对应1.645σ)

        应用场景:
        - Kalman信号偏离 > threshold → 均值回归信号
        - 预警系统阈值配置

        参数:
        - confidence: 置信水平，默认0.95

        返回:
        - float: 最优信号阈值
        """
        print("\n[6/6] 估计信号偏离阈值...")
        returns = self.returns.values
        n = len(returns)

        if n < 50:
            print("   ⚠️ 数据不足50个观测值，使用默认阈值 1.5")
            self.calibrated['signal_threshold'] = 1.5
            self.diagnostics['signal_threshold'] = {'method': 'fallback', 'n': n}
            return 1.5

        # 计算标准化残差
        mean = np.mean(returns)
        std = np.std(returns)

        if std < 1e-10:
            print("   ⚠️ 数据标准差接近零，使用默认阈值 1.5")
            self.calibrated['signal_threshold'] = 1.5
            self.diagnostics['signal_threshold'] = {'method': 'zero_std', 'n': n}
            return 1.5

        z_scores = (returns - mean) / std

        # 使用绝对值的分位数作为阈值
        # 95%分位数 → 约1.645σ (正态假设下)
        # 但实际数据可能更肥尾，所以分位数可能更高
        abs_z = np.abs(z_scores)
        threshold_est = np.percentile(abs_z, confidence * 100)

        # 计算不同置信水平下的阈值（提供选项）
        thresholds_at_levels = {}
        for level in [0.90, 0.95, 0.99]:
            thresholds_at_levels[level] = np.percentile(abs_z, level * 100)

        self.calibrated['signal_threshold'] = threshold_est
        self.diagnostics['signal_threshold'] = {
            'method': f'分位数估计 ({confidence*100:.0f}%)',
            'thresholds_at_levels': thresholds_at_levels,
            'n': n
        }

        print(f"   threshold_opt = {threshold_est:.2f} (vs 交易经验默认 1.5)")
        print(f"   多级阈值: 90%={thresholds_at_levels[0.90]:.2f}, "
              f"95%={thresholds_at_levels[0.95]:.2f}, "
              f"99%={thresholds_at_levels[0.99]:.2f}")

        return threshold_est

    def diagnose_garch_persistence(self, garch_results=None):
        """
        校验GARCH参数的持久性约束: alpha + beta < 1

        方法: 提取GARCH模型参数并检查
        - alpha + beta < 1: 平稳条件
        - alpha + beta ≈ 1: IGARCH（无限持久）
        - alpha + beta > 1: 非平稳（爆炸性方差）

        参数:
        - garch_results: dict, GARCH模型结果（可选）
            若提供，则校验实际参数；否则使用校准估计值

        返回:
        - dict: 持久性诊断结果
        """
        print("\n[GARCH持久性校验]")
        diagnostics = {}

        if garch_results is not None:
            for model_name, result in garch_results.items():
                try:
                    params = result.params
                    # 识别alpha和beta参数名
                    alpha_keys = [k for k in params.index if 'alpha' in k.lower()]
                    beta_keys = [k for k in params.index if 'beta' in k.lower()]

                    alpha_sum = sum(params[k] for k in alpha_keys) if alpha_keys else 0
                    beta_sum = sum(params[k] for k in beta_keys) if beta_keys else 0
                    persistence = alpha_sum + beta_sum

                    diagnostics[model_name] = {
                        'alpha': alpha_sum,
                        'beta': beta_sum,
                        'persistence': persistence,
                        'stationary': persistence < 1,
                        'half_life': -np.log(2) / np.log(persistence) if 0 < persistence < 1 else np.inf
                    }

                    status = "平稳" if persistence < 1 else ("IGARCH" if abs(persistence - 1) < 0.01 else "非平稳")
                    print(f"  {model_name}: α+β = {persistence:.4f} ({status})")
                    if persistence < 1 and persistence > 0:
                        print(f"    半衰期 = {-np.log(2)/np.log(persistence):.1f} 日")

                except Exception as e:
                    diagnostics[model_name] = {'error': str(e)}
        else:
            print("  未提供GARCH结果，跳过持久性校验")
            print("  建议: 传入VolatilityModeler.results进行诊断")

        self.diagnostics['garch_persistence'] = diagnostics
        return diagnostics

    def get_calibration_config(self):
        """
        生成校准后的配置字典，可直接用于替代硬编码默认值

        返回:
        - dict: 校准参数配置
        """
        config = {}

        if 'ewma_lambda' in self.calibrated:
            config['EWMA_LAMBDA'] = self.calibrated['ewma_lambda']

        if 't_df' in self.calibrated:
            config['T_DF'] = self.calibrated['t_df']

        if 'ar_phi' in self.calibrated:
            config['AR_PHI'] = self.calibrated['ar_phi']

        if 'evt_threshold_percentile' in self.calibrated:
            config['EVT_THRESHOLD_PERCENTILE'] = self.calibrated['evt_threshold_percentile']

        if 'kalman_window' in self.calibrated:
            config['KALMAN_WINDOW'] = self.calibrated['kalman_window']

        if 'signal_threshold' in self.calibrated:
            config['SIGNAL_THRESHOLD'] = self.calibrated['signal_threshold']

        return config

    def print_calibration_report(self):
        """打印详细校准报告"""
        print("\n" + "="*60)
        print("参数校准详细报告")
        print("="*60)

        defaults = {
            'ewma_lambda': 0.94,
            't_df': 5.0,
            'ar_phi': 0.98,
            'evt_threshold_percentile': 0.95,
            'kalman_window': 60,
            'signal_threshold': 1.5
        }

        for param, calibrated_value in self.calibrated.items():
            default = defaults.get(param, 'N/A')
            diag = self.diagnostics.get(param, {})
            method = diag.get('method', 'N/A')
            ci = diag.get('ci', 'N/A')

            print(f"\n  {param}:")
            print(f"    默认值: {default}")
            print(f"    校准值: {calibrated_value:.4f}")
            print(f"    变化: {calibrated_value - default if isinstance(default, (int, float)) else 'N/A'}")
            print(f"    方法: {method}")
            print(f"    置信区间: {ci}")

            # 参数特定诊断
            if param == 't_df' and 'tail_diagnosis' in diag:
                print(f"    肥尾诊断: {diag['tail_diagnosis']}")
            elif param == 'ar_phi' and 'persist_diagnosis' in diag:
                print(f"    持久性诊断: {diag['persist_diagnosis']}")
            elif param == 'ar_phi' and 'significant' in diag:
                print(f"    统计显著性: {diag['significant']} (p={diag.get('p_value', 'N/A')})")

        print("\n" + "="*60)
