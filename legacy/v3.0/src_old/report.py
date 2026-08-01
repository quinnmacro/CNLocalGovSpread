"""
战略报告生成模块 - 自动生成可执行的交易建议和风险预警
"""


def generate_strategic_report(
    winner_model,
    vol_modeler,
    clean_data,
    smoothed_spread,
    signal_deviation,
    winner_volatility,
    evt_var,
    evt_es=None
):
    """
    生成战略分析报告

    参数:
    - evt_es: Expected Shortfall值 (可选)
    """
    print("\n" + "="*80)
    print(" " * 20 + "中国地方债利差战略分析报告")
    print("="*80)

    # ---------- 第一部分: 模型选择结果 ----------
    print("\n【一、模型锦标赛结果】")
    print("-" * 80)
    print(f"✓ 获胜模型: {winner_model}")
    print(f"  - AIC: {vol_modeler.ic_scores[winner_model]['AIC']:.2f}")
    print(f"  - BIC: {vol_modeler.ic_scores[winner_model]['BIC']:.2f}")

    # 对比所有模型
    print("\n  模型对比:")
    for model_name, scores in vol_modeler.ic_scores.items():
        winner_mark = "← 🏆" if model_name == winner_model else ""
        print(f"    {model_name:15s}  AIC={scores['AIC']:8.2f}  BIC={scores['BIC']:8.2f}  {winner_mark}")

    print(f"\n  📊 结论: 基于 AIC 准则，{winner_model} 模型最能解释中国地方债利差的动态特征")

    # ---------- 第二部分: 不对称效应检验 ----------
    print("\n【二、波动率不对称效应】")
    print("-" * 80)

    asymmetry_detected = False
    if winner_model == 'EGARCH':
        # P0修复: arch库EGARCH不含显式gamma参数，非对称性通过|z_t|项隐式体现
        # 无法直接提取gamma值，需参考GJR-GARCH结果获取显式非对称性度量
        gjr_gamma = None
        if 'GJR-GARCH' in vol_modeler.results:
            gjr_gamma = vol_modeler.results['GJR-GARCH'].params.get('gamma[1]', 0)
        if gjr_gamma is not None and gjr_gamma > 0.05:
            print(f"  ✓ 参考GJR-GARCH结果: 检测到显著杠杆效应 (γ = {gjr_gamma:.4f})")
            print(f"    → EGARCH通过|z_t|项隐式捕捉非对称性，GJR-GARCH γ={gjr_gamma:.4f}确认该效应")
            asymmetry_detected = True
        else:
            print(f"  ℹ️  EGARCH的非对称性通过|z_t|项隐式体现（arch库不含显式γ参数）")
            print(f"    → 若需显式非对称效应度量，请参考GJR-GARCH模型结果")

    elif winner_model == 'GJR-GARCH':
        gamma = vol_modeler.results['GJR-GARCH'].params.get('gamma[1]', 0)
        if gamma > 0.05:
            print(f"  ✓ 检测到显著的杠杆效应 (γ = {gamma:.4f})")
            print(f"    → 负冲击对波动率的影响比正冲击大 {gamma:.1%}")
            asymmetry_detected = True
        else:
            print(f"  ℹ️  杠杆效应不显著")
    else:
        print(f"  ℹ️  标准 GARCH 模型胜出，未检测到不对称效应")
        print(f"    → 说明正负冲击对波动率的影响较为对称")

    if asymmetry_detected:
        print("\n  🎯 交易含义: 在利差快速扩大时,应预期波动率会超比例上升,需加大对冲力度")

    # ---------- 第三部分: 风险预警 ----------
    print("\n【三、当前风险状况】")
    print("-" * 80)

    current_spread = clean_data['spread'].iloc[-1]
    current_trend = smoothed_spread.iloc[-1]
    deviation_bps = current_spread - current_trend
    deviation_sigma = signal_deviation.iloc[-1]
    current_vol = winner_volatility.iloc[-1]

    print(f"  当前利差水平:        {current_spread:.2f} bps")
    print(f"  卡尔曼趋势水平:      {current_trend:.2f} bps")
    print(f"  偏离程度:            {deviation_bps:+.2f} bps ({deviation_sigma:+.2f}σ)")
    print(f"  当前波动率:          {current_vol:.2f} bps/日")
    print(f"  99% EVT-VaR:        {evt_var:.2f} bps (单日最大风险)")
    if evt_es is not None:
        print(f"  99% EVT-ES:         {evt_es:.2f} bps (尾部平均损失)")

    # 风险等级判定
    print("\n  风险等级判定:")
    if abs(deviation_sigma) > 2.0:
        risk_level = "⚠️ 高风险"
        print(f"    {risk_level}: 利差严重偏离趋势 (>2σ)")
        if deviation_sigma > 0:
            print(f"      → 利差可能被高估,存在较大均值回归压力")
        else:
            print(f"      → 利差可能被低估,警惕信用事件风险")
    elif abs(deviation_sigma) > 1.5:
        risk_level = "⚡ 中等风险"
        print(f"    {risk_level}: 利差偏离趋势 (1.5σ - 2σ)")
        print(f"      → 可考虑建立方向性仓位")
    else:
        risk_level = "✓ 低风险"
        print(f"    {risk_level}: 利差在正常波动范围内 (<1.5σ)")
        print(f"      → 维持中性仓位")

    # 波动率状态
    vol_percentile = (winner_volatility < current_vol).mean()
    print(f"\n  波动率状态: 当前处于历史 {vol_percentile:.1%} 分位")
    if vol_percentile > 0.90:
        print(f"    ⚠️  危机模式: 波动率处于极高水平 (>90%分位)")
        print(f"      → 建议降低杠杆,增加现金储备")
    elif vol_percentile > 0.75:
        print(f"    ⚡ 高波动期: 市场不确定性上升")
        print(f"      → 谨慎加仓,控制风险敞口")
    else:
        print(f"    ✓ 正常波动期")

    # ---------- 第四部分: 行动建议 ----------
    print("\n【四、行动建议】")
    print("-" * 80)

    print("  基于当前分析,建议:")
    print()

    # 方向性建议
    if deviation_sigma > 1.5:
        print("  1. 方向性策略: 🔴 做空利差 (预期收窄)")
        print(f"     - 入场点: {current_spread:.2f} bps")
        print(f"     - 目标价: {current_trend:.2f} bps (回归趋势)")
        print(f"     - 止损点: {current_spread + evt_var:.2f} bps (当前+VaR)")
    elif deviation_sigma < -1.5:
        print("  1. 方向性策略: 🟢 做多利差 (预期扩大)")
        print(f"     - 入场点: {current_spread:.2f} bps")
        print(f"     - 目标价: {current_trend:.2f} bps (回归趋势)")
        print(f"     - 止损点: {max(0, current_spread - evt_var):.2f} bps (当前-VaR)")
    else:
        print("  1. 方向性策略: ⚪ 中性观望")
        print(f"     - 当前利差在合理区间,等待更明确信号")

    # 风险管理建议
    print(f"\n  2. 风险管理:")
    print(f"     - 单日 VaR 限额: {evt_var:.2f} bps")
    print(f"     - 建议仓位规模: 假设风险预算为 R bps,则最大名义敞口 = R / {evt_var:.2f}")
    if vol_percentile > 0.90:
        print(f"     - ⚠️  当前高波动环境,建议将仓位削减至正常水平的 50%-70%")

    # 监控指标
    print(f"\n  3. 关键监控指标:")
    print(f"     - 偏离度: 当前 {deviation_sigma:+.2f}σ → 警戒线 ±1.5σ, 止损线 ±2.5σ")
    print(f"     - 波动率: 当前 {current_vol:.2f} → 上升 20% 以上需重新评估风险敞口")
    print(f"     - 趋势: 若卡尔曼趋势突破 {current_trend + current_vol*2:.2f} bps,说明市场regime切换")

    print("\n" + "="*80)
    print(" " * 30 + "报告完成")
    print("="*80)
