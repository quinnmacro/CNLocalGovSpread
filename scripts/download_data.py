#!/usr/bin/env python3
"""
Wind 数据下载脚本 — QuinnMacro 地方债利差分析

用法:
    # 下载地方债利差 (默认)
    python scripts/download_data.py

    # 指定日期范围
    python scripts/download_data.py --start 2020-01-01 --end 2026-08-01

    # 同时下载信用利差对比数据
    python scripts/download_data.py --credit

    # 指定 Wind Python 路径 (非标准安装)
    python scripts/download_data.py --wind-path "/custom/path/to/wind"

    # 增量更新 (只下载新数据)
    python scripts/download_data.py --incremental

功能:
    - 下载地方债利差数据 (综合/5Y/10Y/30Y) — Wind EDB M0017142~5
    - 可选下载信用利差对比数据 (企业债/中票 AAA)
    - 增量更新: 检测已有 CSV 最新日期，只拉取新数据
    - 自动检测 macOS/Windows Wind 路径
    - 数据验证 + 摘要统计
    - 保存到 data/ 目录
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def _ensure_data_dir() -> Path:
    """Create data directory if it doesn't exist."""
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def _detect_last_date(csv_path: Path) -> str | None:
    """Detect the last date in an existing CSV for incremental download."""
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"], usecols=["date"])
        if df.empty:
            return None
        last = df["date"].max()
        # Next business day
        next_day = last + pd.offsets.BDay(1)
        return next_day.strftime("%Y-%m-%d")
    except Exception:
        return None


def download_spread_data(
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    output_path: str = "data/local_gov_spread.csv",
    incremental: bool = False,
    wind_path: str | None = None,
) -> pd.DataFrame | None:
    """
    下载地方债信用利差数据

    Parameters
    ----------
    start_date : 起始日期 (YYYY-MM-DD)
    end_date : 结束日期, 默认今天
    output_path : 输出 CSV 路径
    incremental : 增量更新模式 — 检测已有 CSV 并从最新日期后一天开始
    wind_path : 自定义 Wind Python 路径

    Returns
    -------
    DataFrame with columns [date, spread_all, spread_5y, spread_10y, spread_30y]
    """
    from src.core.wind_client import WindClient, DEFAULT_SPREAD_CODES

    output = Path(output_path)
    if not output.is_absolute():
        output = ROOT / output

    # Incremental mode: detect last date from existing CSV
    if incremental:
        last_date = _detect_last_date(output)
        if last_date:
            print(f"  增量模式: 已有数据到 {last_date}, 从该日期开始下载")
            start_date = last_date
        else:
            print("  增量模式: 未找到已有数据，使用全量下载")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  下载地方债信用利差数据")
    print(f"  日期范围: {start_date} → {end_date}")
    print(f"  指标数量: {len(DEFAULT_SPREAD_CODES)}")
    print(f"{'='*60}")

    try:
        with WindClient(wind_path=wind_path) as client:
            df = client.fetch_edb(
                codes=DEFAULT_SPREAD_CODES,
                start_date=start_date,
                end_date=end_date,
                fill_method="Previous",
            )
    except ImportError as exc:
        print(f"\n✗ Wind Python API 未安装:")
        print(f"  {exc}")
        print(f"\n  请确保:")
        print(f"  1. Wind 终端已安装并运行")
        print(f"  2. Wind Python API 已添加到 PATH")
        print(f"  macOS 默认路径: /Applications/Wind API.app/Contents/python")
        return None
    except Exception as exc:
        print(f"\n✗ 数据下载失败: {exc}")
        return None

    if df is None or df.empty:
        print("✗ 未获取到数据")
        return None

    # Merge with existing data in incremental mode
    if incremental and output.exists():
        print("\n  合并已有数据...")
        existing = pd.read_csv(output, parse_dates=["date"])
        # Remove overlapping dates
        existing = existing[existing["date"] < df["date"].min()]
        df = pd.concat([existing, df], ignore_index=True)
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # Save
    _ensure_data_dir()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"\n{'='*60}")
    print(f"  ✓ 数据已保存到 {output.relative_to(ROOT)}")
    print(f"  ✓ {len(df)} 行 × {len(df.columns)} 列")
    print(f"  ✓ 日期范围: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"{'='*60}")

    # Data summary
    spread_cols = [c for c in df.columns if c.startswith("spread_")]
    if spread_cols:
        print("\n  数据摘要:")
        for col in spread_cols:
            s = df[col].dropna()
            print(f"    {col:15s}: 均值={s.mean():.2f}  标准差={s.std():.2f}  "
                  f"最小={s.min():.2f}  最大={s.max():.2f}  最新={s.iloc[-1]:.2f}")

    return df


def download_credit_spread_data(
    start_date: str = "2018-01-01",
    end_date: str | None = None,
    output_path: str = "data/credit_spread.csv",
    wind_path: str | None = None,
) -> pd.DataFrame | None:
    """
    下载信用利差对比数据 (企业债/中票 AAA 各期限)

    注: 需要在 wind_client.py 中配置实际的 Wind EDB 代码。
    当前 CREDIT_SPREAD_CODES 为空占位，需填入实际代码后使用。
    """
    from src.core.wind_client import WindClient, CREDIT_SPREAD_CODES

    if not CREDIT_SPREAD_CODES:
        print("\n⚠️  未配置信用利差 EDB 代码")
        print("  请编辑 src/core/wind_client.py 中的 CREDIT_SPREAD_CODES")
        print("  填入 Wind EDB 实际代码后重新运行")
        return None

    output = Path(output_path)
    if not output.is_absolute():
        output = ROOT / output

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  下载信用利差对比数据")
    print(f"  日期范围: {start_date} → {end_date}")
    print(f"  指标数量: {len(CREDIT_SPREAD_CODES)}")
    print(f"{'='*60}")

    try:
        with WindClient(wind_path=wind_path) as client:
            df = client.fetch_edb(
                codes=CREDIT_SPREAD_CODES,
                start_date=start_date,
                end_date=end_date,
                fill_method="Previous",
            )
    except Exception as exc:
        print(f"\n✗ 信用利差数据下载失败: {exc}")
        return None

    if df is not None and not df.empty:
        _ensure_data_dir()
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)

        print(f"\n  ✓ 数据已保存到 {output.relative_to(ROOT)}")
        print(f"  ✓ {len(df)} 行 × {len(df.columns)} 列")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="QuinnMacro 地方债利差数据下载器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_data.py                          # 全量下载
  python scripts/download_data.py --start 2024-01-01       # 指定起始日期
  python scripts/download_data.py --incremental            # 增量更新
  python scripts/download_data.py --credit                 # 含信用利差
        """,
    )
    parser.add_argument("--start", default="2018-01-01",
                        help="开始日期 (YYYY-MM-DD, 默认: 2018-01-01)")
    parser.add_argument("--end", default=None,
                        help="结束日期 (YYYY-MM-DD, 默认: 今天)")
    parser.add_argument("--output", default="data/local_gov_spread.csv",
                        help="输出文件路径 (默认: data/local_gov_spread.csv)")
    parser.add_argument("--credit", action="store_true",
                        help="同时下载信用利差对比数据")
    parser.add_argument("--incremental", action="store_true",
                        help="增量更新: 检测已有 CSV 并从最新日期后下载")
    parser.add_argument("--wind-path", default=None,
                        help="自定义 Wind Python API 路径")

    args = parser.parse_args()

    print("QuinnMacro 地方债利差数据下载器 v4.0")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Download local gov spread
    df = download_spread_data(
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        incremental=args.incremental,
        wind_path=args.wind_path,
    )

    # Optionally download credit spread
    if args.credit:
        download_credit_spread_data(
            start_date=args.start,
            end_date=args.end,
            wind_path=args.wind_path,
        )

    if df is not None:
        print(f"\n✓ 下载完成!")
    else:
        print(f"\n✗ 下载未完成，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
