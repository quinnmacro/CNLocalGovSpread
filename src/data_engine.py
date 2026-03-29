"""
数据引擎模块 - 支持 Wind EDB 和 Mock 数据

关键功能:
1. 自动处理 Wind EDB 的假期数据缺失
2. MAD 方法剔除极端异常值（比 Z-Score 更稳健）
3. 统一输出格式：DatetimeIndex + 'spread' 列
"""

import numpy as np
import pandas as pd


class DataEngine:
    """
    数据引擎类 - 支持 Wind EDB 和 Mock 数据

    关键功能:
    1. 自动处理 Wind EDB 的假期数据缺失
    2. MAD 方法剔除极端异常值（比 Z-Score 更稳健）
    3. 统一输出格式：DatetimeIndex + 'spread' 列
    """

    def __init__(self, config):
        self.config = config
        self._raw_data = None
        self._clean_data = None

    def load_data(self):
        """加载数据 - 生产环境需接入 Wind API"""
        if self.config['SOURCE'] == 'WIND_EDB':
            # 生产环境代码示例（需要 Wind Python API）:
            # from WindPy import w
            # w.start()
            # data = w.edb(self.config['TICKER'], self.config['START_DATE'], self.config['END_DATE'])
            # self._raw_data = pd.DataFrame(data.Data, index=data.Times, columns=['spread'])
            raise NotImplementedError("请在生产环境配置 Wind API")

        else:  # MOCK 数据
            # 模拟一个真实的地方债利差时间序列：
            # - 均值回归特征 (Mean Reversion)
            # - 波动率聚集 (Volatility Clustering)
            # - 偶尔的尖峰 (Fat Tails)
            np.random.seed(42)
            dates = pd.date_range(self.config['START_DATE'], self.config['END_DATE'], freq='B')
            n = len(dates)

            # 使用 AR(1) + GARCH(1,1) 过程生成模拟数据
            spread = np.zeros(n)
            spread[0] = 100  # 初始利差 100 bps
            volatility = np.zeros(n)
            volatility[0] = 10

            # 这些参数是根据实际地方债市场经验设定的：
            # - phi = 0.98: 高度持久性（地方债利差是慢变量）
            # - omega 很小：长期波动率稳定
            # - alpha + beta 接近 1：波动率聚集效应明显
            phi = 0.98  # AR(1) 系数 - 高持久性
            mu = 100    # 长期均值 100 bps
            omega = 0.5 # GARCH 常数项
            alpha = 0.15 # ARCH 效应
            beta = 0.80  # GARCH 效应

            for t in range(1, n):
                # GARCH(1,1) 波动率更新
                volatility[t] = np.sqrt(omega + alpha * (spread[t-1] - mu)**2 + beta * volatility[t-1]**2)

                # AR(1) 过程 + 随机冲击
                shock = np.random.standard_t(df=5) * volatility[t]  # 使用 t 分布制造肥尾
                spread[t] = mu + phi * (spread[t-1] - mu) + shock

                # 偶尔添加跳跃（模拟政策冲击或信用事件）
                if np.random.rand() < 0.01:  # 1% 概率发生跳跃
                    spread[t] += np.random.choice([-1, 1]) * np.random.uniform(20, 40)

            self._raw_data = pd.DataFrame({'spread': spread}, index=dates)
            print(f"✓ 已生成 {len(dates)} 个交易日的模拟数据")

        return self._raw_data

    def clean_data(self):
        """
        数据清洗 - 使用 MAD (Median Absolute Deviation) 处理异常值

        为什么用 MAD 而不是标准差？
        - Wind EDB 的数据经常有「硬伤」：比如某天突然出现 999 或 -999 的占位符
        - 标准差对这种极端值敏感，会误判正常波动
        - MAD 基于中位数，对离群点有抵抗力（Robust Estimator）
        """
        if self._raw_data is None:
            raise ValueError("请先调用 load_data()")

        df = self._raw_data.copy()

        # Step 1: 处理缺失值（Wind 在节假日会返回 NaN）
        # 这里用 forward fill 是因为利差是慢变量，昨天的值是今天的最佳估计
        df['spread'] = df['spread'].ffill().bfill()

        # Step 2: MAD 异常值检测
        median = df['spread'].median()
        mad = np.median(np.abs(df['spread'] - median))

        # 这个 1.4826 是什么鬼？
        # 答：它是让 MAD 在正态分布下等价于标准差的调整因子
        # 但我们的数据不是正态分布（有肥尾），所以这只是个近似
        threshold = self.config['MAD_THRESHOLD']
        modified_z_score = 0.6745 * (df['spread'] - median) / mad

        outliers = np.abs(modified_z_score) > threshold
        if outliers.sum() > 0:
            print(f"⚠️  检测到 {outliers.sum()} 个异常值（MAD 阈值 = {threshold}）")
            # 用中位数替换异常值（保守做法，避免引入偏差）
            df.loc[outliers, 'spread'] = median

        self._clean_data = df
        print(f"✓ 数据清洗完成，最终样本量: {len(df)}")
        return self._clean_data

    def get_returns(self):
        """计算利差变化（一阶差分）- GARCH 模型的输入"""
        if self._clean_data is None:
            raise ValueError("请先调用 clean_data()")

        # 为什么用差分而不是百分比收益率？
        # 因为利差本身就是绝对值（bps），不是价格
        # 100 bps -> 105 bps 的波动意义和 200 bps -> 205 bps 一样
        returns = self._clean_data['spread'].diff().dropna()
        return returns
