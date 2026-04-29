"""
市场状态仪表模块 - 多指标联动实时状态监控

功能:
1. 多指标评分: 利差定位、波动率状态、VaR突破、信号偏离、趋势动量
2. 加权融合: 数据驱动的权重分配，综合风险评分
3. 仪表可视化: Plotly 交互式多段仪表盘
4. 指标联动雷达图: 展示指标间关联强度
5. 滚动状态时间线: 帆海模型捕捉状态演变
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MarketStatusGauge:
    """
    多指标联动市场状态仪表

    融合5个维度指标生成综合市场状态评分:
    - 利差定位 (Spread Position): 当前利差 vs 历史分布
    - 波动率状态 (Volatility Regime): 当前波动率 vs GARCH条件波动率
    - 风险突破 (VaR Breach): 当前变化 vs EVT-VaR阈值
    - 信号偏离 (Signal Deviation): Kalman滤波信号偏离度
    - 趋势动量 (Trend Momentum): 近期趋势方向与强度
    """

    # 指标权重 (基于各指标对风险预测的贡献度)
    DEFAULT_WEIGHTS = {
        'spread_position': 0.20,
        'volatility_regime': 0.25,
        'var_breach': 0.25,
        'signal_deviation': 0.15,
        'trend_momentum': 0.15
    }

    # 状态等级定义
    STATUS_LEVELS = {
        'safe': {'label': '安全', 'color': '#22c55e', 'range': (0, 20)},
        'watch': {'label': '关注', 'color': '#3b82f6', 'range': (20, 40)},
        'caution': {'label': '警戒', 'color': '#f59e0b', 'range': (40, 60)},
        'warning': {'label': '预警', 'color': '#f97316', 'range': (60, 80)},
        'danger': {'label': '危险', 'color': '#ef4444', 'range': (80, 100)},
    }

    def __init__(self, clean_data, returns, smoothed=None, deviation=None,
                 vol_modeler=None, evt=None, weights=None):
        self.clean_data = clean_data
        self.returns = returns
        self.smoothed = smoothed
        self.deviation = deviation
        self.vol_modeler = vol_modeler
        self.evt = evt
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._indicator_scores = None
        self._composite_score = None

    def calculate_indicator_scores(self):
        """
        计算各维度指标评分 (0-100)

        每个指标归一化到0-100:
        - 0 = 最安全状态
        - 100 = 最危险状态

        返回:
            dict: 各指标评分及详情
        """
        scores = {}
        spread = self.clean_data['spread']

        # 1. 利差定位评分
        # 当前利差相对于历史分布的位置
        # 使用百分位排名而非Z-score，更直观
        current_spread = spread.iloc[-1]
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        # Z-score -> 0-100: 使用双侧累积概率映射
        # z=0 → 50 (中性), z>2 → ~97 (危险), z<-2 → ~2 (安全但低估风险)
        if z_score >= 0:
            spread_score = min(100, stats.norm.cdf(z_score) * 100)
        else:
            # 负Z-score表示利差偏低，但也有风险(可能反弹)
            spread_score = min(100, max(0, 50 - abs(z_score) * 10))

        scores['spread_position'] = {
            'score': spread_score,
            'z_score': z_score,
            'current': current_spread,
            'mean': spread_mean,
            'std': spread_std,
            'percentile': (spread < current_spread).sum() / len(spread) * 100
        }

        # 2. 波动率状态评分
        if self.vol_modeler is not None:
            try:
                winner = self.vol_modeler.run_tournament()
                winner_vol = self.vol_modeler.get_conditional_volatility(winner)
                current_vol = winner_vol.iloc[-1]
                vol_mean = winner_vol.mean()
                vol_p90 = winner_vol.quantile(0.90)

                # 波动率评分: 当前波动率在历史分布中的位置
                vol_ratio = current_vol / vol_mean if vol_mean > 0 else 1
                if vol_ratio <= 0.8:
                    vol_score = 10  # 低波动，安全
                elif vol_ratio <= 1.2:
                    vol_score = 30  # 正常波动
                elif vol_ratio <= 1.5:
                    vol_score = 50  # 波动偏高
                elif vol_ratio <= 2.0:
                    vol_score = 75  # 高波动
                else:
                    vol_score = 95  # 极端波动

                # 检测波动率状态 (Regime)
                regime_info = {}
                if hasattr(self.vol_modeler, 'regime_detector') and self.vol_modeler.regime_detector is not None:
                    try:
                        regime_result = self.vol_modeler.regime_detector.detect_regimes(winner_vol)
                        if regime_result is not None:
                            current_regime = regime_result.iloc[-1]
                            regime_info = {
                                'current_regime': current_regime,
                                'regime_counts': regime_result.value_counts().to_dict()
                            }
                    except Exception:
                        pass

                scores['volatility_regime'] = {
                    'score': vol_score,
                    'current_vol': current_vol,
                    'mean_vol': vol_mean,
                    'p90_vol': vol_p90,
                    'ratio': vol_ratio,
                    'regime': regime_info
                }
            except Exception:
                scores['volatility_regime'] = {
                    'score': 50,  # 默认中性
                    'current_vol': None,
                    'mean_vol': None,
                    'ratio': None,
                    'regime': {}
                }
        else:
            scores['volatility_regime'] = {
                'score': 50,
                'current_vol': None,
                'mean_vol': None,
                'ratio': None,
                'regime': {}
            }

        # 3. VaR突破评分
        if self.evt is not None and hasattr(self.evt, 'var') and self.evt.var is not None:
            var_value = self.evt.var
            current_return = self.returns.iloc[-1]

            # VaR突破概率评分
            # 当前变化接近VaR时评分越高
            breach_ratio = current_return / var_value if var_value != 0 else 0

            if breach_ratio < 0.5:
                var_score = 5   # 远低于VaR
            elif breach_ratio < 0.8:
                var_score = 20  # 安全范围
            elif breach_ratio < 1.0:
                var_score = 50  # 接近VaR
            elif breach_ratio < 1.2:
                var_score = 80  # 超过VaR
            else:
                var_score = 95  # 远超VaR

            # 计算ES/VaR比率 (尾部厚度指标)
            es_value = getattr(self.evt, 'es', None)
            es_var_ratio = abs(es_value) / abs(var_value) if (es_value and var_value) else None

            scores['var_breach'] = {
                'score': var_score,
                'var': var_value,
                'es': es_value,
                'current_return': current_return,
                'breach_ratio': breach_ratio,
                'es_var_ratio': es_var_ratio
            }
        else:
            # 使用经验分位数作为fallback
            emp_var = self.returns.quantile(0.99)
            current_return = self.returns.iloc[-1]
            scores['var_breach'] = {
                'score': 50,
                'var': emp_var,
                'es': None,
                'current_return': current_return,
                'breach_ratio': None,
                'es_var_ratio': None
            }

        # 4. 信号偏离评分
        if self.deviation is not None:
            current_deviation = self.deviation.iloc[-1]
            abs_deviation = abs(current_deviation)

            # 偏离度 -> 评分映射
            if abs_deviation < 0.5:
                dev_score = 10  # 微弱偏离
            elif abs_deviation < 1.0:
                dev_score = 25  # 小幅偏离
            elif abs_deviation < 1.5:
                dev_score = 45  # 中度偏离
            elif abs_deviation < 2.0:
                dev_score = 70  # 强偏离
            elif abs_deviation < 2.5:
                dev_score = 85  # 极强偏离
            else:
                dev_score = 95  # 极端偏离

            scores['signal_deviation'] = {
                'score': dev_score,
                'current_deviation': current_deviation,
                'abs_deviation': abs_deviation,
                'direction': '高估' if current_deviation > 0 else '低估',
                'signal_type': '做空信号' if current_deviation > 1.5 else
                              '做多信号' if current_deviation < -1.5 else '中性'
            }
        else:
            scores['signal_deviation'] = {
                'score': 50,
                'current_deviation': None,
                'abs_deviation': None,
                'direction': None,
                'signal_type': None
            }

        # 5. 趋势动量评分
        # 近期趋势方向和强度
        if len(spread) >= 60:
            recent_mean = spread.iloc[-20:].mean()
            prev_mean = spread.iloc[-60:-20].mean()
            trend_delta = recent_mean - prev_mean

            # 趋势强度 (相对标准差)
            trend_strength = abs(trend_delta) / spread_std if spread_std > 0 else 0

            # 扩大利差趋势 → 更高风险评分
            # 收窄利差趋势 → 较低风险评分
            if trend_delta > 0:
                # 利差扩大 (不利方向)
                if trend_strength < 0.3:
                    trend_score = 40  # 缓慢扩大
                elif trend_strength < 0.5:
                    trend_score = 55  # 中速扩大
                elif trend_strength < 1.0:
                    trend_score = 75  # 快速扩大
                else:
                    trend_score = 90  # 急剧扩大
            else:
                # 利差收窄 (有利方向，但收窄过快也可能有问题)
                if trend_strength < 0.3:
                    trend_score = 15  # 缓慢收窄，理想
                elif trend_strength < 0.5:
                    trend_score = 25  # 中速收窄
                elif trend_strength < 1.0:
                    trend_score = 35  # 快速收窄，关注反弹
                else:
                    trend_score = 50  # 急剧收窄，可能反弹

            scores['trend_momentum'] = {
                'score': trend_score,
                'trend_delta': trend_delta,
                'trend_strength': trend_strength,
                'direction': '扩大' if trend_delta > 0 else '收窄',
                'recent_mean': recent_mean,
                'prev_mean': prev_mean
            }
        else:
            scores['trend_momentum'] = {
                'score': 50,
                'trend_delta': None,
                'trend_strength': None,
                'direction': None,
                'recent_mean': None,
                'prev_mean': None
            }

        self._indicator_scores = scores
        return scores

    def calculate_composite_score(self):
        """
        计算加权融合综合评分

        返回:
            dict: 综合评分及状态判断
        """
        if self._indicator_scores is None:
            self.calculate_indicator_scores()

        # 加权求和
        composite = 0
        total_weight = 0
        for indicator, weight in self.weights.items():
            if indicator in self._indicator_scores:
                composite += self._indicator_scores[indicator]['score'] * weight
                total_weight += weight

        composite = composite / total_weight if total_weight > 0 else 50
        composite = min(100, max(0, composite))

        # 状态判断
        status = 'safe'
        for level_name, level_info in self.STATUS_LEVELS.items():
            lo, hi = level_info['range']
            if lo <= composite < hi:
                status = level_name
                break
        if composite >= 80:
            status = 'danger'

        # 极端指标检测: 任一指标>=90触发升级
        for indicator, info in self._indicator_scores.items():
            if info['score'] >= 90:
                if status in ('safe', 'watch', 'caution'):
                    status = 'warning'
                elif status == 'warning':
                    status = 'danger'

        self._composite_score = {
            'score': composite,
            'status': status,
            'label': self.STATUS_LEVELS[status]['label'],
            'color': self.STATUS_LEVELS[status]['color'],
            'indicator_scores': {k: v['score'] for k, v in self._indicator_scores.items()},
            'weights': self.weights
        }

        return self._composite_score

    def get_market_status(self):
        """获取市场状态摘要"""
        if self._composite_score is None:
            self.calculate_composite_score()

        return self._composite_score

    def calculate_indicator_correlation(self, window=60):
        """
        计算指标间滚动相关性 (指标联动分析)

        返回:
            dict: 指标间相关系数矩阵
        """
        spread = self.clean_data['spread']
        returns = self.returns

        if len(spread) < window:
            return {}

        # 计算各指标的滚动时间序列
        indicators_ts = {}

        # 利差变化
        indicators_ts['spread_change'] = returns.rolling(window).std()

        # 波动率指标 (使用简单滚动标准差作为proxy)
        indicators_ts['volatility'] = returns.rolling(window).std()

        # 利差水平偏离
        indicators_ts['spread_level'] = ((spread - spread.rolling(window).mean()) /
                                          spread.rolling(window).std())

        # 趋势指标
        indicators_ts['trend'] = spread.rolling(20).mean() - spread.rolling(60).mean()

        # 构建DataFrame并计算相关性
        df = pd.DataFrame(indicators_ts).dropna()
        if len(df) < 10:
            return {}

        corr_matrix = df.corr()

        # 转换为嵌套dict格式
        correlation = {}
        for col1 in corr_matrix.columns:
            correlation[col1] = {}
            for col2 in corr_matrix.columns:
                correlation[col1][col2] = corr_matrix.loc[col1, col2]

        return correlation

    def plot_status_gauge(self, theme='light'):
        """
        绘制多段市场状态仪表 (主仪表 + 5个子仪表)

        参数:
            theme: 'light' 或 'dark'

        返回:
            plotly Figure 对象
        """
        if self._composite_score is None:
            self.calculate_composite_score()

        composite = self._composite_score
        scores = self._indicator_scores

        # 指标名称映射 (中文)
        indicator_names = {
            'spread_position': '利差定位',
            'volatility_regime': '波动率状态',
            'var_breach': 'VaR突破',
            'signal_deviation': '信号偏离',
            'trend_momentum': '趋势动量'
        }

        # 子仪表颜色映射
        def score_to_color(score):
            if score < 20:
                return '#22c55e'
            elif score < 40:
                return '#3b82f6'
            elif score < 60:
                return '#f59e0b'
            elif score < 80:
                return '#f97316'
            else:
                return '#ef4444'

        # 创建2行布局: 主仪表在上，5个子仪表在下
        fig = make_subplots(
            rows=2, cols=5,
            specs=[[{'type': 'indicator'}] * 5] +
                  [[{'type': 'indicator'}] * 5],
            vertical_spacing=0.35
        )

        # 主仪表 (居中显示在第一行中间)
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=composite['score'],
            domain={'x': [0.1, 0.9], 'y': [0.55, 1]},
            title={
                'text': f"市场状态: <b>{composite['label']}</b>",
                'font': {'size': 18}
            },
            delta={'reference': 30, 'increasing': {'color': '#ef4444'},
                   'decreasing': {'color': '#22c55e'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1,
                         'tickcolor': 'darkgray', 'tickfont': {'size': 10}},
                'bar': {'color': composite['color'], 'thickness': 0.6},
                'bgcolor': 'rgba(255,255,255,0.1)',
                'borderwidth': 2,
                'bordercolor': 'gray',
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(34,197,94,0.3)'},
                    {'range': [20, 40], 'color': 'rgba(59,130,246,0.3)'},
                    {'range': [40, 60], 'color': 'rgba(245,158,11,0.3)'},
                    {'range': [60, 80], 'color': 'rgba(249,115,22,0.3)'},
                    {'range': [80, 100], 'color': 'rgba(239,68,68,0.3)'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 3},
                    'thickness': 0.75,
                    'value': composite['score']
                }
            }
        ), row=1, col=3)

        # 5个子仪表 (第二行)
        for i, (key, name) in enumerate(indicator_names.items()):
            score_val = scores[key]['score']
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=score_val,
                title={'text': name, 'font': {'size': 12}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 0.5,
                             'tickcolor': 'lightgray', 'tickfont': {'size': 8}},
                    'bar': {'color': score_to_color(score_val), 'thickness': 0.7},
                    'bgcolor': 'rgba(255,255,255,0.05)',
                    'borderwidth': 1,
                    'bordercolor': 'lightgray',
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(34,197,94,0.15)'},
                        {'range': [40, 70], 'color': 'rgba(245,158,11,0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239,68,68,0.15)'}
                    ]
                }
            ), row=2, col=i + 1)

        # 主题配置
        if theme == 'dark':
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC',
                height=450,
                margin=dict(l=20, r=20, t=50, b=20)
            )
        else:
            fig.update_layout(
                template='none',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#0F172A',
                height=450,
                margin=dict(l=20, r=20, t=50, b=20)
            )

        return fig

    def plot_indicator_linkage(self, theme='light'):
        """
        绘制指标联动雷达图

        展示各指标评分和联动强度

        参数:
            theme: 'light' 或 'dark'

        返回:
            plotly Figure 对象
        """
        if self._composite_score is None:
            self.calculate_composite_score()

        indicator_names = {
            'spread_position': '利差定位',
            'volatility_regime': '波动率状态',
            'var_breach': 'VaR突破',
            'signal_deviation': '信号偏离',
            'trend_momentum': '趋势动量'
        }

        categories = [indicator_names[k] for k in indicator_names]
        values = [self._indicator_scores[k]['score'] for k in indicator_names]
        # 雷达图需要闭合: 添加第一个值到末尾
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        # 安全基准线 (各指标理想状态)
        baseline_values = [15, 15, 15, 15, 15]
        baseline_closed = baseline_values + [baseline_values[0]]

        fig = go.Figure()

        # 当前状态
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(59,130,246,0.25)',
            line=dict(color='#3b82f6', width=2),
            name='当前状态'
        ))

        # 安全基准
        fig.add_trace(go.Scatterpolar(
            r=baseline_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(34,197,94,0.1)',
            line=dict(color='#22c55e', width=1, dash='dot'),
            name='安全基准'
        ))

        # 警戒线 (50分)
        caution_values = [50, 50, 50, 50, 50]
        caution_closed = caution_values + [caution_values[0]]
        fig.add_trace(go.Scatterpolar(
            r=caution_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(245,158,11,0.08)',
            line=dict(color='#f59e0b', width=1, dash='dash'),
            name='警戒线'
        ))

        if theme == 'dark':
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        tickfont=dict(color='#F8FAFC', size=9),
                        gridcolor='rgba(71,85,105,0.3)'
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='#F8FAFC', size=11),
                        gridcolor='rgba(71,85,105,0.3)'
                    )
                ),
                height=400,
                showlegend=True,
                legend=dict(font=dict(color='#F8FAFC')),
                font=dict(color='#F8FAFC')
            )
        else:
            fig.update_layout(
                template='none',
                paper_bgcolor='rgba(0,0,0,0)',
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        tickfont=dict(color='#0F172A', size=9),
                        gridcolor='rgba(226,232,240,0.8)'
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='#0F172A', size=11),
                        gridcolor='rgba(226,232,240,0.8)'
                    )
                ),
                height=400,
                showlegend=True,
                font=dict(color='#0F172A')
            )

        return fig

    def plot_indicator_timeline(self, window=60, theme='light'):
        """
        绘制滚动指标评分时间线

        展示各指标评分随时间的变化，捕捉市场状态演变

        参数:
            window: 滚动窗口大小
            theme: 'light' 或 'dark'

        返回:
            plotly Figure 对象
        """
        spread = self.clean_data['spread']
        returns = self.returns
        n = len(spread)

        if n < window * 2:
            # 数据不足，返回空图表
            fig = go.Figure()
            fig.add_annotation(
                text="数据不足，无法计算滚动指标时间线",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            if theme == 'dark':
                fig.update_layout(template='plotly_dark', height=400,
                                  paper_bgcolor='rgba(0,0,0,0)')
            else:
                fig.update_layout(template='none', height=400,
                                  paper_bgcolor='rgba(0,0,0,0)')
            return fig

        # 计算滚动指标评分序列
        # 注意: returns 比 spread 少1行(diff), 需要按日期对齐
        spread_dates = spread.index[window:]

        # 利差定位 (滚动百分位)
        spread_scores = []
        for dt in spread_dates:
            hist = spread.loc[:dt]
            current = spread.loc[dt]
            pct = (hist < current).sum() / len(hist) * 100
            spread_scores.append(pct)

        # 波动率状态 (滚动波动率比率)
        vol_scores = []
        rolling_std = returns.rolling(window).std()
        mean_std = rolling_std.mean()
        for dt in spread_dates:
            if dt in rolling_std.index and not pd.isna(rolling_std.loc[dt]):
                current_std = rolling_std.loc[dt]
                ratio = current_std / mean_std if mean_std > 0 else 1
            else:
                ratio = 1.0
            if ratio <= 0.8:
                vol_scores.append(10)
            elif ratio <= 1.2:
                vol_scores.append(30)
            elif ratio <= 1.5:
                vol_scores.append(50)
            elif ratio <= 2.0:
                vol_scores.append(75)
            else:
                vol_scores.append(95)

        # 信号偏离 (如果有Kalman数据)
        dev_scores = []
        if self.deviation is not None:
            deviation_aligned = self.deviation.reindex(spread_dates)
            for dt in spread_dates:
                if dt in deviation_aligned.index and not pd.isna(deviation_aligned.loc[dt]):
                    abs_dev = abs(deviation_aligned.loc[dt])
                    if abs_dev < 0.5:
                        dev_scores.append(10)
                    elif abs_dev < 1.0:
                        dev_scores.append(25)
                    elif abs_dev < 1.5:
                        dev_scores.append(45)
                    elif abs_dev < 2.0:
                        dev_scores.append(70)
                    elif abs_dev < 2.5:
                        dev_scores.append(85)
                    else:
                        dev_scores.append(95)
                else:
                    dev_scores.append(50)
        else:
            dev_scores = [50] * len(spread_dates)

        # 趋势动量 (滚动趋势)
        trend_scores = []
        for dt in spread_dates:
            loc = spread.index.get_loc(dt)
            if loc >= 60:
                recent_mean = spread.iloc[loc-20:loc].mean()
                prev_mean = spread.iloc[loc-60:loc-20].mean()
                delta = recent_mean - prev_mean
                std = spread.iloc[:loc].std()
                strength = abs(delta) / std if std > 0 else 0
                if delta > 0:
                    if strength < 0.3:
                        trend_scores.append(40)
                    elif strength < 0.5:
                        trend_scores.append(55)
                    else:
                        trend_scores.append(75)
                else:
                    if strength < 0.3:
                        trend_scores.append(15)
                    elif strength < 0.5:
                        trend_scores.append(25)
                    else:
                        trend_scores.append(35)
            else:
                trend_scores.append(50)

        # 确保所有序列长度一致
        min_len = min(len(spread_dates), len(spread_scores), len(vol_scores),
                      len(dev_scores), len(trend_scores))
        dates = spread_dates[:min_len]
        spread_scores = spread_scores[:min_len]
        vol_scores = vol_scores[:min_len]
        dev_scores = dev_scores[:min_len]
        trend_scores = trend_scores[:min_len]

        # 综合评分时间线
        w = self.weights
        composite_ts = (
            np.array(spread_scores) * w['spread_position'] +
            np.array(vol_scores) * w['volatility_regime'] +
            np.array(dev_scores) * w['signal_deviation'] +
            np.array(trend_scores) * w['trend_momentum']
        )
        # VaR breach 使用固定50 (无滚动VaR计算)
        composite_ts += 50 * w['var_breach']

        fig = go.Figure()

        # 综合评分 (粗线)
        fig.add_trace(go.Scatter(
            x=dates, y=composite_ts,
            mode='lines', name='综合评分',
            line=dict(color='#1d4ed8', width=2.5)
        ))

        # 各指标评分
        colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b']
        indicator_data = [
            (spread_scores, '利差定位'),
            (vol_scores, '波动率状态'),
            (dev_scores, '信号偏离'),
            (trend_scores, '趋势动量')
        ]
        for (data, name), color in zip(indicator_data, colors):
            fig.add_trace(go.Scatter(
                x=dates, y=data,
                mode='lines', name=name,
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.7
            ))

        # 风险等级分隔线
        for threshold, label, color in [(20, '安全', '#22c55e'),
                                          (40, '关注', '#3b82f6'),
                                          (60, '警戒', '#f59e0b'),
                                          (80, '预警', '#f97316')]:
            fig.add_hline(y=threshold, line_dash='dash',
                          line_color=color, line_width=1,
                          annotation_text=label,
                          annotation_font=dict(color=color, size=9),
                          annotation_position='right')

        fig.update_layout(
            title='市场状态演变时间线',
            xaxis_title='日期',
            yaxis_title='风险评分 (0-100)',
            hovermode='x unified',
            height=450,
            legend=dict(orientation='h', yanchor='bottom',
                        y=1.02, xanchor='right', x=1)
        )

        if theme == 'dark':
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8FAFC'
            )
            fig.update_xaxes(gridcolor='rgba(71,85,105,0.3)')
            fig.update_yaxes(gridcolor='rgba(71,85,105,0.3)')
        else:
            fig.update_layout(
                template='none',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#0F172A'
            )
            fig.update_xaxes(gridcolor='rgba(226,232,240,0.8)')
            fig.update_yaxes(gridcolor='rgba(226,232,240,0.8)')

        return fig
