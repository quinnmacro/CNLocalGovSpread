"""
样式系统 - Dashboard 主题和组件样式

功能:
1. 深色/浅色主题切换
2. 卡片式组件
3. 图表主题配置
"""

import streamlit as st

# ============================================================================
# 主题 CSS
# ============================================================================

LIGHT_THEME = """
<style>
    :root {
        --primary: #1E3A5F;
        --secondary: #667eea;
        --accent: #764ba2;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-main: #FAFAFA;
        --bg-card: #FFFFFF;
        --bg-secondary: #F1F5F9;
        --text-primary: #1E1E1E;
        --text-secondary: #64748B;
        --border-color: #E2E8F0;
    }

    .stApp {
        background-color: var(--bg-main);
    }

    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 8px;
    }

    .metric-card-success {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    }

    .metric-card-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }

    .metric-card-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }

    .metric-card-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }

    /* Metric 组件样式 */
    .stMetric {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 14px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* Container 样式 */
    .stContainer {
        border-radius: 12px;
        background: var(--bg-card);
    }

    /* Tab 样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* Expander 样式 */
    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border-radius: 8px;
    }

    /* 数据框样式 */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* 标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }

    .sub-title {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--primary);
        margin: 1rem 0;
    }

    /* 预警样式 */
    .alert-danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 4px solid var(--danger);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }

    .alert-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid var(--warning);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }

    .alert-success {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid var(--success);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }

    .alert-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }

    /* 页脚样式 */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
    }

    .footer a {
        color: var(--secondary);
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }
</style>
"""

DARK_THEME = """
<style>
    :root {
        --primary: #60A5FA;
        --secondary: #818CF8;
        --accent: #A78BFA;
        --success: #34D399;
        --warning: #FBBF24;
        --danger: #F87171;
        --bg-main: #0E1117;
        --bg-card: #1E1E1E;
        --bg-secondary: #262730;
        --text-primary: #FAFAFA;
        --text-secondary: #9CA3AF;
        --border-color: #374151;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
    }

    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 8px;
    }

    .metric-card-success {
        background: linear-gradient(135deg, #34D399 0%, #10B981 100%);
    }

    .metric-card-warning {
        background: linear-gradient(135deg, #FBBF24 0%, #F59E0B 100%);
    }

    .metric-card-danger {
        background: linear-gradient(135deg, #F87171 0%, #EF4444 100%);
    }

    .metric-card-info {
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
    }

    /* Metric 组件样式 */
    .stMetric {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 14px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* Container 样式 */
    .stContainer {
        border-radius: 12px;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
    }

    /* Tab 样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-secondary);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        color: var(--text-secondary);
    }

    .stTabs [aria-selected="true"] {
        color: var(--text-primary);
        background: var(--bg-card);
    }

    /* Expander 样式 */
    .streamlit-expanderHeader {
        background: var(--bg-secondary);
        border-radius: 8px;
        color: var(--text-primary);
    }

    /* 数据框样式 */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        background: var(--bg-card);
    }

    /* 标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }

    .sub-title {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--primary);
        margin: 1rem 0;
    }

    /* 预警样式 */
    .alert-danger {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border-left: 4px solid var(--danger);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #fecaca;
    }

    .alert-warning {
        background: linear-gradient(135deg, #451a03 0%, #78350f 100%);
        border-left: 4px solid var(--warning);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #fde68a;
    }

    .alert-success {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border-left: 4px solid var(--success);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #86efac;
    }

    .alert-info {
        background: linear-gradient(135deg, #0c4a6e 0%, #155e75 100%);
        border-left: 4px solid #60A5FA;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #93c5fd;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: var(--bg-card);
        border-right: 1px solid var(--border-color);
    }

    /* 输入框样式 */
    .stTextInput input, .stSelectbox select, .stNumberInput input {
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
    }

    /* 滑块样式 */
    .stSlider {
        background: transparent;
    }

    /* 按钮样式 */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%);
    }

    /* 页脚样式 */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
    }

    .footer a {
        color: var(--secondary);
        text-decoration: none;
    }

    .footer a:hover {
        text-decoration: underline;
    }

    /* Plotly 图表深色背景 */
    .js-plotly-plot .plotly .modebar {
        background: transparent !important;
    }
</style>
"""


# ============================================================================
# Plotly 图表主题配置
# ============================================================================

def get_plotly_theme(theme='light'):
    """获取 Plotly 图表主题配置"""
    if theme == 'dark':
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#FAFAFA'},
            'xaxis': {'gridcolor': '#374151', 'linecolor': '#374151'},
            'yaxis': {'gridcolor': '#374151', 'linecolor': '#374151'},
        }
    else:
        return {
            'template': 'plotly_white',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'font': {'color': '#1E1E1E'},
            'xaxis': {'gridcolor': '#E2E8F0', 'linecolor': '#E2E8F0'},
            'yaxis': {'gridcolor': '#E2E8F0', 'linecolor': '#E2E8F0'},
        }


# ============================================================================
# 卡片组件
# ============================================================================

def metric_card(title, value, delta=None, status="normal"):
    """
    渲染带样式的指标卡片

    参数:
        title: 卡片标题
        value: 主要数值
        delta: 变化值（可选）
        status: 状态 (normal, success, warning, danger, info)
    """
    status_class = {
        'normal': '',
        'success': 'metric-card-success',
        'warning': 'metric-card-warning',
        'danger': 'metric-card-danger',
        'info': 'metric-card-info'
    }.get(status, '')

    delta_html = f'<div style="font-size: 0.9rem; opacity: 0.9;">{delta}</div>' if delta else ''

    html = f"""
    <div class="metric-card {status_class}">
        <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 4px;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 700;">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def alert_box(message, level='info'):
    """
    渲染预警框

    参数:
        message: 预警消息
        level: 预警级别 (info, success, warning, danger)
    """
    level_class = {
        'info': 'alert-info',
        'success': 'alert-success',
        'warning': 'alert-warning',
        'danger': 'alert-danger'
    }.get(level, 'alert-info')

    icon = {
        'info': 'ℹ️',
        'success': '✓',
        'warning': '⚠️',
        'danger': '🔴'
    }.get(level, 'ℹ️')

    html = f"""
    <div class="{level_class}">
        <strong>{icon}</strong> {message}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def section_header(title, icon=None):
    """
    渲染章节标题

    参数:
        title: 标题文本
        icon: 可选图标
    """
    display_title = f"{icon} {title}" if icon else title
    st.markdown(f'<div class="section-title">{display_title}</div>', unsafe_allow_html=True)


# ============================================================================
# 主题应用
# ============================================================================

def apply_theme(theme='light'):
    """
    应用主题样式

    参数:
        theme: 'light' 或 'dark'
    """
    if theme == 'dark':
        st.markdown(DARK_THEME, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_THEME, unsafe_allow_html=True)


def get_theme_toggle():
    """
    创建主题切换按钮

    返回:
        当前主题 ('light' 或 'dark')
    """
    # 初始化 session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

    # 创建切换按钮
    col1, col2 = st.columns([3, 1])
    with col2:
        is_dark = st.toggle("🌙 深色模式", value=(st.session_state.theme == 'dark'))

    # 更新主题
    theme = 'dark' if is_dark else 'light'
    st.session_state.theme = theme

    return theme


# ============================================================================
# 布局组件
# ============================================================================

def render_page_header(title, subtitle=None):
    """
    渲染页面标题

    参数:
        title: 主标题
        subtitle: 副标题（可选）
    """
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sub-title">{subtitle}</div>', unsafe_allow_html=True)


def render_footer(version='1.4.0', author='Quinn Liu', github=None, linkedin=None):
    """
    渲染页脚

    参数:
        version: 版本号
        author: 作者名
        github: GitHub 链接
        linkedin: LinkedIn 链接
    """
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
