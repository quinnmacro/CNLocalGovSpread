"""
样式系统 - 现代化深色主题 Dashboard
专业金融分析界面设计
"""

import streamlit as st

# ============================================================================
# 现代化深色主题 (默认)
# ============================================================================

MODERN_DARK_THEME = """
<style>
    /* 全局变量 */
    :root {
        --primary: #3B82F6;
        --primary-light: #60A5FA;
        --secondary: #8B5CF6;
        --accent: #06B6D4;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --bg-main: #0F172A;
        --bg-card: #1E293B;
        --bg-card-hover: #334155;
        --bg-secondary: #1E293B;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --border-color: #334155;
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }

    /* 主背景 */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
        background-attachment: fixed;
    }

    /* Plotly 图表透明背景 */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }

    /* 顶部导航栏 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(30, 41, 59, 0.5);
        padding: 8px 12px;
        border-radius: 12px;
        margin-bottom: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.95rem;
        color: var(--text-secondary);
        background: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        color: var(--text-primary);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }

    /* Metric 卡片 */
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        border-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.2);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #F8FAFC 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 卡片容器 */
    .card-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: var(--shadow);
        backdrop-filter: blur(10px);
    }

    /* 标题样式 */
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }

    .sub-title {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-bottom: 0.75rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, var(--primary), transparent) 1;
    }

    /* 预警样式 */
    .alert-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        border-left: 3px solid var(--danger);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        color: #FCA5A5;
        font-size: 0.9rem;
        backdrop-filter: blur(5px);
    }

    .alert-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
        border-left: 3px solid var(--warning);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        color: #FCD34D;
        font-size: 0.9rem;
        backdrop-filter: blur(5px);
    }

    .alert-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%);
        border-left: 3px solid var(--success);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        color: #6EE7B7;
        font-size: 0.9rem;
        backdrop-filter: blur(5px);
    }

    .alert-info {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%);
        border-left: 3px solid var(--primary);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        color: #93C5FD;
        font-size: 0.9rem;
        backdrop-filter: blur(5px);
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-right: 1px solid var(--border-color);
    }

    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 8px;
    }

    /* 输入控件 */
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background: rgba(51, 65, 85, 0.5) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }

    /* 按钮 */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }

    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }

    /* 滑块 */
    .stSlider {
        padding: 8px 0;
    }

    .stSlider [data-testid="stSlider"] {
        background: transparent;
    }

    /* 数据表格 */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    .stDataFrame table {
        background: rgba(30, 41, 59, 0.8) !important;
    }

    .stDataFrame thead th {
        background: rgba(51, 65, 85, 0.8) !important;
        color: var(--text-primary) !important;
    }

    .stDataFrame tbody td {
        color: var(--text-primary) !important;
    }

    /* 分隔线 */
    hr {
        border-color: var(--border-color);
        margin: 12px 0;
    }

    /* 页脚 */
    .footer {
        text-align: center;
        padding: 1rem;
        color: var(--text-muted);
        font-size: 0.85rem;
        border-top: 1px solid var(--border-color);
        margin-top: 1.5rem;
    }

    .footer a {
        color: var(--primary-light);
        text-decoration: none;
        transition: color 0.2s ease;
    }

    .footer a:hover {
        color: var(--accent);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(51, 65, 85, 0.3);
        border-radius: 8px;
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }

    /* 容器边框 */
    .stContainer:has(> .element-container) {
        border-radius: 12px;
    }

    /* 状态加载器 */
    .stStatus {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }

    /* 微动画 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .element-container {
        animation: fadeIn 0.3s ease-out;
    }
</style>
"""

# ============================================================================
# 浅色主题 (专业简洁风格)
# ============================================================================

MODERN_LIGHT_THEME = """
<style>
    :root {
        --primary: #2563EB;
        --primary-light: #3B82F6;
        --secondary: #7C3AED;
        --accent: #0891B2;
        --success: #059669;
        --warning: #D97706;
        --danger: #DC2626;
        --bg-main: #FFFFFF;
        --bg-card: #FFFFFF;
        --bg-secondary: #F8FAFC;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --border-color: #E2E8F0;
        --shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    .stApp {
        background: #FFFFFF;
    }

    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }

    /* Tab导航 - 简洁风格 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #F1F5F9;
        padding: 4px;
        border-radius: 10px;
        margin-bottom: 16px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 500;
        color: var(--text-secondary);
        background: transparent;
        border: none;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(37, 99, 235, 0.08);
        color: var(--primary);
    }

    .stTabs [aria-selected="true"] {
        background: #FFFFFF;
        color: var(--primary) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        font-weight: 600;
    }

    /* Metric卡片 - 干净简洁 */
    .stMetric {
        background: #FAFBFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        box-shadow: none;
        transition: all 0.2s ease;
    }

    .stMetric:hover {
        border-color: var(--primary);
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* 标题 - 专业蓝 */
    .main-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-bottom: 0.75rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #E2E8F0;
    }

    /* 预警样式 - 柔和配色 */
    .alert-danger {
        background: #FEF2F2;
        border-left: 3px solid var(--danger);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        color: #B91C1C;
        font-size: 0.9rem;
    }

    .alert-warning {
        background: #FFFBEB;
        border-left: 3px solid var(--warning);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        color: #B45309;
        font-size: 0.9rem;
    }

    .alert-success {
        background: #ECFDF5;
        border-left: 3px solid var(--success);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        color: #047857;
        font-size: 0.9rem;
    }

    .alert-info {
        background: #EFF6FF;
        border-left: 3px solid var(--primary);
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        color: #1D4ED8;
        font-size: 0.9rem;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: #FAFBFC;
        border-right: 1px solid var(--border-color);
    }

    /* 按钮 */
    .stButton button[kind="primary"] {
        background: var(--primary);
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    .stButton button[kind="primary"]:hover {
        background: #1D4ED8;
    }

    /* 数据表格 */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    .stDataFrame table {
        background: #FFFFFF !important;
    }

    .stDataFrame thead th {
        background: #F8FAFC !important;
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    /* 页脚 */
    .footer {
        text-align: center;
        padding: 1rem;
        color: var(--text-muted);
        font-size: 0.85rem;
        border-top: 1px solid var(--border-color);
        margin-top: 1.5rem;
    }

    .footer a {
        color: var(--primary);
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #F8FAFC;
        border-radius: 8px;
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }

    /* 输入控件 */
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background: #FFFFFF;
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }

    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    /* 分隔线 */
    hr {
        border-color: var(--border-color);
        margin: 10px 0;
    }
</style>
"""


# ============================================================================
# 组件函数
# ============================================================================

def metric_card(title, value, delta=None, status="normal"):
    """渲染带样式的指标卡片"""
    status_colors = {
        'normal': 'var(--primary)',
        'success': 'var(--success)',
        'warning': 'var(--warning)',
        'danger': 'var(--danger)',
        'info': 'var(--accent)'
    }

    delta_html = f'<div style="font-size: 0.85rem; opacity: 0.8;">{delta}</div>' if delta else ''
    color = status_colors.get(status, status_colors['normal'])

    html = f"""
    <div style="background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                border: 1px solid {color}40;
                border-radius: 12px;
                padding: 16px;
                margin: 4px 0;">
        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 4px;">{title}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: {color};">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def alert_box(message, level='info'):
    """渲染预警框"""
    level_class = {
        'info': 'alert-info',
        'success': 'alert-success',
        'warning': 'alert-warning',
        'danger': 'alert-danger'
    }.get(level, 'alert-info')

    html = f'<div class="{level_class}">{message}</div>'
    st.markdown(html, unsafe_allow_html=True)


def section_header(title, icon=None):
    """渲染章节标题"""
    display_title = f"{icon} {title}" if icon else title
    st.markdown(f'<div class="section-title">{display_title}</div>', unsafe_allow_html=True)


def apply_theme(theme='dark'):
    """应用主题样式"""
    if theme == 'light':
        st.markdown(MODERN_LIGHT_THEME, unsafe_allow_html=True)
    else:
        st.markdown(MODERN_DARK_THEME, unsafe_allow_html=True)


def get_theme_toggle():
    """创建主题切换按钮"""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'  # 默认深色主题

    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        is_light = st.toggle("☀️ 浅色", value=(st.session_state.theme == 'light'))

    theme = 'light' if is_light else 'dark'
    st.session_state.theme = theme
    return theme


def render_page_header(title, subtitle=None):
    """渲染页面标题"""
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sub-title">{subtitle}</div>', unsafe_allow_html=True)


def render_footer(version='2.1.0', author='Quinn Liu', github=None, linkedin=None):
    """渲染页脚"""
    links = []
    if github:
        links.append(f'<a href="{github}" target="_blank">GitHub</a>')
    if linkedin:
        links.append(f'<a href="{linkedin}" target="_blank">LinkedIn</a>')

    links_html = ' | '.join(links)
    if links_html:
        links_html = f' | {links_html}'

    st.markdown(f"""
    <div class="footer">
        <strong>CNLocalGovSpread</strong> v{version} | Author: {author}{links_html}
    </div>
    """, unsafe_allow_html=True)
