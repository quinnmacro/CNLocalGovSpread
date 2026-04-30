"""
极值理论风险分析模块 - 量化尾部风险

方法: Peaks Over Threshold (POT)
分布: Generalized Pareto Distribution (GPD)
"""

import numpy as np
from scipy import stats
import plotly.graph_objects as go


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
        self.es = None  # Expected Shortfall
        self.hill_estimator = None  # Hill估计量结果

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

        # P0修复: 检查 exceedances 是否为空
        if len(exceedances) == 0:
            print(f"⚠️  警告: 没有数据超过阈值 {self.threshold:.2f}，无法拟合 GPD")
            self.gpd_params = None
            return

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
        VaR_α = u + (σ/ξ) * [ ((n/N_u) * (1-α))^(-ξ) - 1 ]

        其中:
        - u: 阈值
        - σ, ξ: GPD 参数
        - α: 置信水平（如 0.99）
        - n: 总样本数
        - N_u: 超过阈值的样本数
        """
        if self.gpd_params is None:
            # Fallback: 使用经验分位数
            self.var = self.returns.quantile(self.confidence)
            print(f"\n使用经验分位数: {self.confidence*100}% VaR = {self.var:.4f}")
            return self.var

        shape = self.gpd_params['shape']
        scale = self.gpd_params['scale']

        # 形状参数合理性检查
        if abs(shape) > 1.0:
            print(f"⚠️ 形状参数过大 (ξ={shape:.4f})，使用经验分位数")
            self.var = self.returns.quantile(self.confidence)
            return self.var

        # P0修复: 计算样本比例因子 n/N_u
        # 正确的EVT-VaR公式: VaR = u + (σ/ξ) * [((n/N_u) * (1-α))^(-ξ) - 1]
        # 其中 n 是总样本数，N_u 是超过阈值的样本数
        exceedances = self.returns[self.returns > self.threshold]
        n_u = len(exceedances)  # 超过阈值的样本数
        n_total = len(self.returns)  # 总样本数

        if n_u == 0:
            print(f"⚠️ 没有超过阈值的样本，使用经验分位数")
            self.var = self.returns.quantile(self.confidence)
            return self.var

        # EVT-VaR 公式
        if abs(shape) < 1e-6:  # shape ≈ 0 时，用指数分布公式
            self.var = self.threshold - scale * np.log((n_total / n_u) * (1 - self.confidence))
        else:
            # 安全计算，防止溢出 - 使用正确的样本比例因子
            exponent_term = ((n_total / n_u) * (1 - self.confidence)) ** (-shape)

            # 检查指数项是否溢出
            if np.isinf(exponent_term) or np.isnan(exponent_term) or exponent_term > 1e10:
                print(f"⚠️ EVT计算溢出，使用经验分位数")
                self.var = self.returns.quantile(self.confidence)
                return self.var

            var_estimate = self.threshold + (scale / shape) * (exponent_term - 1)

            # VaR 上限保护：不应超过数据范围的 10 倍
            max_reasonable_var = self.returns.max() * 10
            min_reasonable_var = self.returns.min()

            if var_estimate > max_reasonable_var:
                print(f"⚠️ VaR估计值异常大 ({var_estimate:.4f})，使用上限值")
                self.var = max_reasonable_var
            elif var_estimate < min_reasonable_var:
                self.var = min_reasonable_var
            else:
                self.var = var_estimate

        print(f"\n" + "="*60)
        print(f"🎯 EVT-VaR ({self.confidence*100}% 置信水平)")
        print(f"   最大日损失预期: {self.var:.4f}")
        print(f"   解读: 在 100 个交易日中，最坏的那一天利差扩大不超过此值")
        print("="*60)

        return self.var

    def get_tail_index(self):
        """返回尾部指数（Heavy-tail Index）"""
        if self.gpd_params is None:
            return None
        # 尾部指数 = 1/ξ（ξ 越大，尾部越重）
        return 1 / self.gpd_params['shape'] if self.gpd_params['shape'] > 0 else np.inf

    def calculate_es(self):
        """
        计算 Expected Shortfall (Conditional VaR)

        ES = E[Loss | Loss > VaR]
        也称为 CVaR (Conditional VaR) 或 AVaR (Average VaR)

        对于 GPD 分布，ES 有解析解:
        ES_α = VaR_α + (σ + ξ * (VaR_α - u)) / (1 - ξ)

        条件: ξ < 1 (否则ES不存在)
        """
        if self.var is None:
            raise ValueError("请先调用 calculate_var()")

        print("\n" + "="*60)
        print("计算 Expected Shortfall (CVaR)")
        print("="*60)

        # VaR 合理性检查
        if np.isinf(self.var) or np.isnan(self.var):
            print(f"⚠️ VaR 无效，使用经验方法计算 ES")
            exceed_var = self.returns[self.returns > self.returns.quantile(0.99)]
            self.es = exceed_var.mean() if len(exceed_var) > 0 else self.returns.max()
            print(f"经验 ES = {self.es:.4f}")
            return self.es

        if self.gpd_params is None:
            # 经验 ES: 超过 VaR 的值的平均
            exceed_var = self.returns[self.returns > self.var]
            if len(exceed_var) > 0:
                self.es = exceed_var.mean()
            else:
                self.es = self.var  # fallback

            print(f"使用经验方法: {self.confidence*100}% ES = {self.es:.4f}")
            return self.es

        shape = self.gpd_params['shape']
        scale = self.gpd_params['scale']

        if shape >= 1:
            # ES 不存在，使用经验方法
            print(f"⚠️ ξ = {shape:.4f} >= 1, Expected Shortfall 理论值不存在")
            exceed_var = self.returns[self.returns > self.var]
            self.es = exceed_var.mean() if len(exceed_var) > 0 else self.var * 1.2
            print(f"使用经验方法: ES = {self.es:.4f}")
            return self.es

        # GPD-based ES 公式
        es_estimate = self.var + (scale + shape * (self.var - self.threshold)) / (1 - shape)

        # ES 合理性检查：ES 应该 >= VaR
        if np.isinf(es_estimate) or np.isnan(es_estimate):
            print(f"⚠️ ES 计算溢出，使用经验方法")
            exceed_var = self.returns[self.returns > self.var]
            self.es = exceed_var.mean() if len(exceed_var) > 0 else self.var * 1.2
        else:
            # ES 必须 >= VaR
            self.es = max(es_estimate, self.var)

        print(f"🎯 Expected Shortfall ({self.confidence*100}% 置信水平)")
        print(f"   ES = {self.es:.4f}")
        print(f"   VaR = {self.var:.4f}")
        if self.var > 0:
            print(f"   ES/VaR 比率 = {self.es/self.var:.2%}")
        print(f"   解读: 在最坏的1%交易日中,平均损失为 {self.es:.4f}")
        print("="*60)

        return self.es

    def mean_excess_plot(self, min_percentile=0.5, max_percentile=0.99, n_points=50):
        """
        P0修复: 绘制Mean Excess Plot（均值超额图）辅助阈值选择

        Mean Excess Function (MEF): e(u) = E[X - u | X > u]
        如果GPD拟合良好，MEF在阈值之上应近似线性

        参数:
        - min_percentile: 最小百分位起点
        - max_percentile: 最大百分位终点
        - n_points: 计算点数

        返回:
        - dict: {'thresholds': array, 'mean_excess': array, 'optimal_threshold': float}
        """
        if self.returns is None:
            raise ValueError("数据未加载")

        data = self.returns.values
        thresholds = np.linspace(
            np.percentile(data, min_percentile * 100),
            np.percentile(data, max_percentile * 100),
            n_points
        )

        mean_excess = []
        for u in thresholds:
            exceedances = data[data > u] - u
            if len(exceedances) > 5:
                mean_excess.append(np.mean(exceedances))
            else:
                mean_excess.append(np.nan)

        mean_excess = np.array(mean_excess)

        # 寻找MEF近似线性的区域起始点（最优阈值）
        # 方法: 找到MEF开始稳定变化的最低阈值
        valid_mask = ~np.isnan(mean_excess)
        if np.sum(valid_mask) > 10:
            valid_thresh = thresholds[valid_mask]
            valid_me = mean_excess[valid_mask]

            # 计算MEF变化率（局部斜率），寻找稳定区域
            diffs = np.diff(valid_me)
            stable_start = 0
            for i in range(len(diffs)):
                # 检查连续5个点的斜率是否相对稳定（方差小）
                if i + 5 <= len(diffs):
                    window_var = np.var(diffs[i:i+5])
                    total_var = np.var(diffs)
                    if window_var < total_var * 0.5:
                        stable_start = i
                        break

            optimal_threshold = valid_thresh[stable_start] if stable_start < len(valid_thresh) else valid_thresh[-1]
        else:
            optimal_threshold = np.percentile(data, 95)

        self.mean_excess_data = {
            'thresholds': thresholds,
            'mean_excess': mean_excess,
            'optimal_threshold': optimal_threshold
        }

        # 绘制交互式图表
        fig = go.Figure()

        # MEF曲线
        fig.add_trace(go.Scatter(
            x=thresholds, y=mean_excess,
            mode='lines+markers',
            name='Mean Excess Function',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=5)
        ))

        # 标记最优阈值
        fig.add_trace(go.Scatter(
            x=[optimal_threshold],
            y=[mean_excess[thresholds == optimal_threshold][0] if optimal_threshold in thresholds else np.nan],
            mode='markers',
            name='推荐阈值',
            marker=dict(color='#FF5722', size=12, symbol='diamond')
        ))

        # 标记当前阈值
        if self.threshold is not None:
            current_me = mean_excess[np.argmin(np.abs(thresholds - self.threshold))]
            fig.add_trace(go.Scatter(
                x=[self.threshold], y=[current_me],
                mode='markers',
                name=f'当前阈值 (P{self.threshold_percentile*100:.0f})',
                marker=dict(color='#4CAF50', size=12, symbol='star')
            ))

        fig.update_layout(
            title='Mean Excess Plot - 阈值选择辅助',
            xaxis_title='阈值 u',
            yaxis_title='E[X - u | X > u]',
            template='plotly_white',
            height=500
        )

        print(f"\n  推荐阈值: {optimal_threshold:.2f} bps")
        print(f"  MEF在该阈值之上近似线性，GPD拟合假设合理")

        return {'fig': fig, 'optimal_threshold': optimal_threshold,
                'thresholds': thresholds, 'mean_excess': mean_excess}

    def estimate_hill(self, k_percentile=0.10):
        """
        使用Hill估计量估计尾部指数

        Hill估计量是尾部指数的非参数估计方法，可交叉验证GPD拟合结果。

        参数:
        - k_percentile: 极值样本占比，默认10%

        返回:
        - hill_tail_index: Hill尾部指数

        公式:
        ξ_Hill = 1/k * Σ[log(X_i / X_{k+1})]
        其中 X_{k+1} 是第(k+1)大的观测值（阈值）
        """
        if self.returns is None:
            raise ValueError("数据未加载")

        # 排序收益率（降序，取正尾部）
        sorted_returns = np.sort(self.returns.values)[::-1]

        # 确定k值（极值样本数）
        k = int(len(sorted_returns) * k_percentile)
        if k < 10:
            k = 10  # 最少10个极值样本
        if k > len(sorted_returns) - 1:
            k = len(sorted_returns) - 1

        # Hill估计量计算
        threshold = sorted_returns[k]
        exceedances = sorted_returns[:k]

        # P0修复: 检查 exceedances 和 threshold 是否有效
        if k == 0 or threshold == 0:
            print(f"⚠️  警告: 无法计算 Hill 估计量 (k={k}, threshold={threshold:.2f})")
            self.hill_estimator = {
                'tail_index': np.inf,
                'shape': np.inf,
                'threshold': threshold,
                'k': k
            }
            return np.inf

        # ξ_Hill = 1/k * Σ[log(X_i / threshold)]
        # P0修复: 添加安全检查，防止 log(0) 或除零
        safe_exceedances = exceedances[exceedances > 0]
        if len(safe_exceedances) == 0:
            print(f"⚠️  警告: 没有有效的正极值样本")
            self.hill_estimator = {
                'tail_index': np.inf,
                'shape': np.inf,
                'threshold': threshold,
                'k': k
            }
            return np.inf

        log_sum = np.sum(np.log(safe_exceedances / threshold))
        hill_xi = log_sum / k if k > 0 else np.inf

        # 尾部指数 = 1/ξ
        # P0修复: Hill估计量仅适用于正xi（重尾分布）
        # 当xi<0时，分布有有限上界（短尾），不应转换为正数
        if hill_xi > 0:
            hill_tail_index = 1 / hill_xi
        elif hill_xi < 0:
            hill_tail_index = -1 / hill_xi  # 负值表示短尾分布
            print(f"  ⚠️  ξ < 0 表示短尾分布（有上界），Hill估计量可能不适用")
        else:
            hill_tail_index = np.inf

        self.hill_estimator = {
            'tail_index': hill_tail_index,
            'shape': hill_xi,
            'threshold': threshold,
            'k': k
        }

        print("\n" + "="*60)
        print("Hill估计量结果")
        print("="*60)
        print(f"  极值样本数 k = {k}")
        print(f"  阈值 = {threshold:.2f} bps")
        print(f"  形状参数 ξ_Hill = {hill_xi:.4f}")
        print(f"  尾部指数 α = {hill_tail_index:.2f}")

        # 与GPD结果对比
        if self.gpd_params is not None:
            gpd_xi = self.gpd_params['shape']
            diff = hill_xi - gpd_xi
            pct_diff = diff / gpd_xi * 100 if gpd_xi != 0 else 0
            print(f"\n  与GPD对比:")
            print(f"    GPD ξ  = {gpd_xi:.4f}")
            print(f"    Hill ξ = {hill_xi:.4f}")
            print(f"    差异   = {diff:.4f} ({pct_diff:.1f}%)")
            if abs(pct_diff) < 20:
                print(f"    ✓ 两种方法结果接近，尾部估计稳健")
            else:
                print(f"    ⚠️  差异较大，建议检查阈值选择")

        print("="*60)

        return hill_tail_index
