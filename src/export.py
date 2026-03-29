"""
数据导出模块 - 将分析结果导出到Excel

功能:
- 多Sheet结构化输出
- 包含原始数据、信号分析、波动率、风险指标
"""

import pandas as pd
from datetime import datetime


def export_to_excel(
    output_path,
    clean_data=None,
    returns=None,
    smoothed_spread=None,
    signal_deviation=None,
    winner_volatility=None,
    winner_model=None,
    evt_var=None,
    evt_es=None,
    config=None
):
    """
    将分析结果导出到Excel文件

    参数:
    - output_path: 输出文件路径
    - clean_data: 清洗后的原始数据
    - returns: 利差变化序列
    - smoothed_spread: 卡尔曼平滑利差
    - signal_deviation: 信号偏离度
    - winner_volatility: 条件波动率
    - winner_model: 获胜模型名称
    - evt_var: VaR值
    - evt_es: ES值
    - config: 配置字典

    返回:
    - output_path: 保存的文件路径
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: 原始数据
        if clean_data is not None:
            clean_data.to_excel(writer, sheet_name='原始数据')

        # Sheet 2: 利差变化
        if returns is not None:
            returns.to_frame('spread_change').to_excel(writer, sheet_name='利差变化')

        # Sheet 3: 信号分析
        if smoothed_spread is not None and signal_deviation is not None:
            signal_df = pd.DataFrame({
                'spread': clean_data['spread'] if clean_data is not None else None,
                'smoothed_trend': smoothed_spread,
                'signal_deviation': signal_deviation
            })
            signal_df.to_excel(writer, sheet_name='信号分析')

        # Sheet 4: 波动率
        if winner_volatility is not None:
            vol_df = winner_volatility.to_frame('conditional_volatility')
            if winner_model:
                vol_df.columns = [f'{winner_model}_volatility']
            vol_df.to_excel(writer, sheet_name='波动率')

        # Sheet 5: 风险指标
        risk_data = {}
        if evt_var is not None:
            risk_data['VaR_99%_(bps)'] = [round(evt_var, 2)]
        if evt_es is not None:
            risk_data['ES_99%_(bps)'] = [round(evt_es, 2)]
        if winner_model:
            risk_data['Winner_Model'] = [winner_model]
        if config:
            risk_data['Data_Source'] = [config.get('SOURCE', 'N/A')]
            risk_data['Start_Date'] = [config.get('START_DATE', 'N/A')]
            risk_data['End_Date'] = [config.get('END_DATE', 'N/A')]
        risk_data['Export_Time'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]

        risk_df = pd.DataFrame(risk_data)
        risk_df.to_excel(writer, sheet_name='风险指标', index=False)

    print(f"✓ 分析结果已导出到: {output_path}")
    return output_path
