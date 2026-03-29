"""
Wind 数据下载脚本

用法:
    python scripts/download_data.py

功能:
    - 下载地方债利差数据 (综合/5Y/10Y/30Y)
    - 保存到 data/local_gov_spread.csv
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置 Wind Python API 路径
wind_path = '/Applications/Wind API.app/Contents/python'
if wind_path not in sys.path:
    sys.path.insert(0, wind_path)

from WindPy import w
import pandas as pd
from datetime import datetime


# Wind EDB 指标代码 - 地方债利差
WIND_INDICATORS = {
    'M0017142': 'spread_all',    # 地方债利差综合
    'M0017143': 'spread_5y',     # 5年期
    'M0017144': 'spread_10y',    # 10年期
    'M0017145': 'spread_30y',    # 30年期
}

# Wind EDB 指标代码 - 信用利差对比
# 注：以下为企业债、中票信用利差指标，需要根据实际Wind EDB代码更新
CREDIT_SPREAD_INDICATORS = {
    # 企业债信用利差 (AAA)
    # 'M00XXXXX': 'credit_corp_aaa',    # 企业债AAA信用利差
    # 'M00XXXXX': 'credit_corp_5y',     # 企业债5Y信用利差
    # 'M00XXXXX': 'credit_corp_10y',    # 企业债10Y信用利差

    # 中票信用利差 (AAA)
    # 'M00XXXXX': 'credit_mtn_aaa',     # 中票AAA信用利差
    # 'M00XXXXX': 'credit_mtn_5y',      # 中票5Y信用利差
    # 'M00XXXXX': 'credit_mtn_10y',     # 中票10Y信用利差
}


def download_local_gov_spread(start_date='2018-01-01', end_date=None, output_path='data/local_gov_spread.csv'):
    """
    下载地方债利差数据

    参数:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期，默认为今天
        output_path: 输出文件路径

    返回:
        DataFrame: 下载的数据
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"正在下载地方债利差数据 ({start_date} ~ {end_date})...")

    # 启动 Wind
    w.start()
    print("✓ Wind API 已连接")

    # 先获取日期
    first_ticker = list(WIND_INDICATORS.keys())[0]
    date_data = w.edb(first_ticker, start_date, end_date)

    if date_data.ErrorCode != 0:
        w.stop()
        raise ValueError(f"Wind 数据获取失败，错误码: {date_data.ErrorCode}")

    df = pd.DataFrame({'date': date_data.Times})
    print(f"✓ 获取到 {len(df)} 个交易日")

    # 下载各期限数据
    for ticker, col_name in WIND_INDICATORS.items():
        print(f"  下载 {col_name} ({ticker})...", end=' ')
        data = w.edb(ticker, start_date, end_date)

        if data.ErrorCode == 0:
            df[col_name] = data.Data[0]
            print(f"✓ {len(data.Data[0])} 条记录")
        else:
            print(f"✗ 错误码: {data.ErrorCode}")

    # 关闭 Wind
    w.stop()
    print("✓ Wind API 已断开")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存到 CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ 数据已保存到 {output_path}")
    print(f"  总计 {len(df)} 行, {len(df.columns)} 列")

    return df


def download_credit_spread(start_date='2018-01-01', end_date=None, output_path='data/credit_spread.csv'):
    """
    下载信用利差数据（企业债、中票等）

    参数:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期，默认为今天
        output_path: 输出文件路径

    返回:
        DataFrame: 下载的数据
    """
    if not CREDIT_SPREAD_INDICATORS:
        print("⚠️ 未配置信用利差指标，请先在脚本中添加 Wind EDB 代码")
        return None

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"正在下载信用利差数据 ({start_date} ~ {end_date})...")

    w.start()
    print("✓ Wind API 已连接")

    first_ticker = list(CREDIT_SPREAD_INDICATORS.keys())[0]
    date_data = w.edb(first_ticker, start_date, end_date)

    if date_data.ErrorCode != 0:
        w.stop()
        raise ValueError(f"Wind 数据获取失败，错误码: {date_data.ErrorCode}")

    df = pd.DataFrame({'date': date_data.Times})
    print(f"✓ 获取到 {len(df)} 个交易日")

    for ticker, col_name in CREDIT_SPREAD_INDICATORS.items():
        print(f"  下载 {col_name} ({ticker})...", end=' ')
        data = w.edb(ticker, start_date, end_date)

        if data.ErrorCode == 0:
            df[col_name] = data.Data[0]
            print(f"✓ {len(data.Data[0])} 条记录")
        else:
            print(f"✗ 错误码: {data.ErrorCode}")

    w.stop()
    print("✓ Wind API 已断开")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ 数据已保存到 {output_path}")

    return df


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='下载债券利差数据')
    parser.add_argument('--start', default='2018-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', default='data/local_gov_spread.csv', help='输出文件路径')
    parser.add_argument('--credit', action='store_true', help='下载信用利差数据')

    args = parser.parse_args()

    if args.credit:
        df = download_credit_spread(
            start_date=args.start,
            end_date=args.end,
            output_path='data/credit_spread.csv'
        )
    else:
        df = download_local_gov_spread(
            start_date=args.start,
            end_date=args.end,
            output_path=args.output
        )

    if df is not None:
        print("\n数据概览:")
        print(df.describe())


if __name__ == '__main__':
    main()
