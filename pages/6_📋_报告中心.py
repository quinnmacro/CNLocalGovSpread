"""
报告中心页面 - 多格式报告生成
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared_state import (
    init_page, render_sidebar, ensure_analysis,
    get_results, safe_metric, render_app_footer,
    generate_report, get_report_history, generate_quick_report,
    render_report_guide, render_page_header
)

# ============================================================================
# 页面初始化
# ============================================================================

theme = init_page(page_title="报告中心", page_icon="📋")
config, run_analysis = render_sidebar()
ensure_analysis(config, run_analysis)

render_page_header("📋 报告中心", "Multi-format Report Generation")

# ============================================================================
# 报告中心内容
# ============================================================================

results = get_results()
if results:
    clean_data = results['clean_data']
    returns = results['returns']
    kalman = results['kalman']
    vol_modeler = results['vol_modeler']
    evt = results['evt']

    # 报告解读指南
    with st.expander("📋 报告解读指南", expanded=False):
        render_report_guide()

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("报告标题", "地方债利差分析报告")
        fmt = st.selectbox("格式", ["HTML", "PDF", "Excel", "PPT"])
        template = st.selectbox("模板", ["professional", "academic", "executive"],
                                format_func=lambda x: {
                                    'professional': '🏢 专业版',
                                    'academic': '📚 学术版',
                                    'executive': '👔 执行版'
                                }.get(x, x))
    with col2:
        sections = st.multiselect(
            "章节",
            ["数据概览", "信号分析", "波动率分析", "风险分析", "交易建议"],
            default=["数据概览", "信号分析", "风险分析"]
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 生成报告", type="primary"):
            with st.spinner("生成中..."):
                path = generate_report(clean_data, returns, kalman, vol_modeler, evt,
                                      title=title, format=fmt, sections=sections,
                                      template=template)
                st.success(f"✓ {path}")
                st.session_state['report'] = path

    with col2:
        if st.button("⚡ 快速报告"):
            path = generate_quick_report(clean_data, returns, kalman, vol_modeler, evt)
            st.success(f"✓ {path}")
            st.session_state['report'] = path

    if 'report' in st.session_state and os.path.exists(st.session_state['report']):
        with open(st.session_state['report'], "rb") as f:
            st.download_button("📥 下载", f, file_name=os.path.basename(st.session_state['report']), type="primary")

    # 历史报告
    st.divider()
    st.markdown("### 📁 历史报告")
    history = get_report_history()
    if history is not None and not history.empty:
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("暂无历史报告")

render_app_footer()