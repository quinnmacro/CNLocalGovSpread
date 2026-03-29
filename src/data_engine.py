"""
数据引擎模块 - 支持 Wind EDB、CSV 和 Mock 数据

关键功能:
1. 多数据源支持: Wind EDB (生产)、CSV (离线)、Mock (测试)
2. 多期限利差: 支持综合/5Y/10Y/30Y 利差数据
3. MAD 方法剔除极端异常值（比 Z-Score 更稳健）
4. 统一输出格式：DatetimeIndex + 'spread' 列
"""

import numpy as np
import pandas as pd
import sys
import os


class DataEngine:
    """
    数据引擎类 - 支持 Wind EDB、CSV 和 Mock 数据

    关键功能:
    1. 多数据源支持: Wind EDB (生产)、CSV (离线)、Mock (测试)
    2. 多期限利差: 支持综合/5Y/10Y/30Y 利差数据
    3. MAD 方法剔除极端异常值（比 Z-Score 更稳健）
    4. 统一输出格式：DatetimeIndex + 'spread' 列
    """

    def __init__(self, config):
        self.config = config
        self._raw_data = None
        self._clean_data = None

    def load_data(self):
        """加载数据 - 支持 Wind EDB、CSV 和 Mock 数据源"""
        source = self.config.get('SOURCE', 'MOCK')

        if source == 'WIND_EDB':
            return self._load_from_wind()
        elif source == 'CSV':
            return self._load_from_csv()
        else:  # MOCK
            return self._generate_mock_data()

    def _load_from_wind(self):
        """从 Wind EDB 获取数据（需要 Wind 终端）"""
        # 配置 Wind Python API 路径
        wind_path = '/Applications/Wind API.app/Contents/python'
        if wind_path not in sys.path:
            sys.path.insert(0, wind_path)

        try:
            from WindPy import w
        except ImportError:
            raise ImportError(
                "Wind Python API 未安装。请确保 Wind 终端已安装，"
                "并将 Python 路径配置到 /Applications/Wind API.app/Contents/python"
            )

        w.start()

        ticker = self.config.get('TICKER', 'M0017142')
        start = self.config.get('START_DATE', '2018-01-01')
        end = self.config.get('END_DATE', '2026-03-29')

        data = w.edb(ticker, start, end)

        if data.ErrorCode != 0:
            w.stop()
            raise ValueError(f"Wind 数据获取失败，错误码: {data.ErrorCode}")

        self._raw_data = pd.DataFrame({
            'spread': data.Data[0]
        }, index=pd.to_datetime(data.Times))

        w.stop()
        print(f"✓ 从 Wind EDB 加载 {len(self._raw_data)} 个交易日数据")
        return self._raw_data

    def _load_from_csv(self):
        """从 CSV 文件加载数据"""
        csv_path = self.config.get('CSV_PATH', 'data/local_gov_spread.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据文件不存在: {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')

        # 支持多列选择
        col = self.config.get('SPREAD_COLUMN', 'spread_all')
        if col not in df.columns:
            available = [c for c in df.columns if c.startswith('spread')]
            raise ValueError(f"列 '{col}' 不存在，可用列: {available}")

        self._raw_data = df[[col]].rename(columns={col: 'spread'})
        print(f"✓ 从 CSV 加载 {len(self._raw_data)} 个交易日数据 ({col})")
        return self._raw_data

    def _generate_mock_data(self):
        """生成模拟数据 - 用于测试和演示"""
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
        threshold = self.config.get('MAD_THRESHOLD', 5.0)

        # P0修复: 检查 MAD 是否为零，避免除零错误
        if mad == 0:
            print("⚠️  MAD = 0，数据无明显离散，跳过异常值检测")
            modified_z_score = pd.Series(0, index=df.index)
        else:
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
