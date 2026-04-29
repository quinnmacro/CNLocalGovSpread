"""
省份聚类热力图模块 - 地方债利差省级差异分析与可视化

功能:
1. 省级利差数据生成 (31省/直辖市/自治区 Mock 数据)
2. 层次聚类 (Hierarchical Clustering) - 基于利差特征向量
3. 聚类热力图 - Dendrogram + 热力图联动可视化
4. 地理分布图 - 省级利差 choropleth 地图
5. 聚类统计 - 各簇利差特征摘要与风险评估

聚类特征向量维度:
- 平均利差水平 (mean_spread)
- 利差波动率 (spread_volatility)
- 偏度 (skewness)
- 与国债利差相关性 (corr_with_national)
- 偏离均值程度 (deviation_from_mean)
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# 中国31省/直辖市/自治区
# ============================================================================

PROVINCES = [
    '北京', '天津', '河北', '山西', '内蒙古',
    '辽宁', '吉林', '黑龙江', '上海', '江苏',
    '浙江', '安徽', '福建', '江西', '山东',
    '河南', '湖北', '湖南', '广东', '广西',
    '海南', '重庆', '四川', '贵州', '云南',
    '西藏', '陕西', '甘肃', '青海', '宁夏',
    '新疆'
]

# 省份简称映射 (用于choropleth)
PROVINCE_SHORT = {
    '北京': '京', '天津': '津', '河北': '冀', '山西': '晋', '内蒙古': '蒙',
    '辽宁': '辽', '吉林': '吉', '黑龙江': '黑', '上海': '沪', '江苏': '苏',
    '浙江': '浙', '安徽': '皖', '福建': '闽', '江西': '赣', '山东': '鲁',
    '河南': '豫', '湖北': '鄂', '湖南': '湘', '广东': '粤', '广西': '桂',
    '海南': '琼', '重庆': '渝', '四川': '川', '贵州': '贵', '云南': '云',
    '西藏': '藏', '陕西': '陕', '甘肃': '甘', '青海': '青', '宁夏': '宁',
    '新疆': '新'
}

# 区域分类 (用于聚类初始种子)
REGION_GROUPS = {
    '东部沿海': ['北京', '天津', '河北', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南'],
    '中部内陆': ['山西', '安徽', '江西', '河南', '湖北', '湖南'],
    '西部开发': ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'],
    '东北老工业': ['辽宁', '吉林', '黑龙江']
}

# 区域利差特征种子 (Mock数据生成基准)
REGION_BASE_PROFILES = {
    '东部沿海': {'mean': 55, 'vol': 8, 'skew': 0.2, 'corr': 0.92},
    '中部内陆': {'mean': 75, 'vol': 12, 'skew': 0.5, 'corr': 0.85},
    '西部开发': {'mean': 95, 'vol': 18, 'skew': 0.8, 'corr': 0.72},
    '东北老工业': {'mean': 85, 'vol': 15, 'skew': 0.6, 'corr': 0.78}
}


# ============================================================================
# 省级聚类分析器
# ============================================================================

class ProvinceClusterMap:
    """
    省级利差聚类分析器

    基于利差特征向量对31省进行层次聚类，
    识别利差模式分组和省级差异结构。
    """

    def __init__(self, province_data=None, n_clusters=4, seed=42):
        """
        参数:
            province_data: DataFrame, 省级利差特征数据
                           列: ['mean_spread', 'spread_volatility', 'skewness',
                                'corr_with_national', 'deviation_from_mean']
                           行: 各省份名
            n_clusters: 聚类数量 (默认4簇, 对应东部/中部/西部/东北)
            seed: 随机种子
        """
        self.n_clusters = n_clusters
        self.seed = seed
        self._province_data = province_data
        self._cluster_labels = None
        self._linkage_matrix = None
        self._cluster_stats = None

        if province_data is None:
            self._province_data = self.generate_mock_province_data()

    def generate_mock_province_data(self):
        """
        生成省级利差Mock数据

        模拟逻辑:
        - 基于区域特征种子生成各省特征
        - 添加省内随机扰动反映个体差异
        - 计算偏离全国均值程度
        """
        np.random.seed(self.seed)

        rows = []
        national_mean = 70  # 全国平均利差 (bps)

        for province in PROVINCES:
            # 确定所属区域
            region = self._get_region(province)
            profile = REGION_BASE_PROFILES[region]

            # 基于区域基准 + 省内扰动
            mean_spread = profile['mean'] + np.random.normal(0, profile['vol'] * 0.3)
            spread_vol = profile['vol'] + abs(np.random.normal(0, 2))
            skewness = profile['skew'] + np.random.normal(0, 0.15)
            corr_with_national = profile['corr'] + np.random.normal(0, 0.05)
            corr_with_national = np.clip(corr_with_national, 0.5, 0.99)
            deviation_from_mean = mean_spread - national_mean

            rows.append({
                'province': province,
                'mean_spread': mean_spread,
                'spread_volatility': spread_vol,
                'skewness': skewness,
                'corr_with_national': corr_with_national,
                'deviation_from_mean': deviation_from_mean
            })

        df = pd.DataFrame(rows).set_index('province')
        return df

    def _get_region(self, province):
        """获取省份所属区域"""
        for region, provinces in REGION_GROUPS.items():
            if province in provinces:
                return region
        return '中部内陆'  # 默认

    def get_feature_matrix(self):
        """获取聚类特征矩阵 (标准化后)"""
        features = [
            'mean_spread', 'spread_volatility', 'skewness',
            'corr_with_national', 'deviation_from_mean'
        ]
        X = self._province_data[features].values
        # Z-score标准化
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        return X_std, features

    def run_clustering(self, method='ward', metric='euclidean'):
        """
        执行层次聚类

        参数:
            method: 联接方法 ('ward', 'complete', 'average', 'single')
            metric: 距离度量 ('euclidean', 'correlation')

        返回:
            cluster_labels: 各省聚类标签 (1..n_clusters)
        """
        X_std, features = self.get_feature_matrix()

        # Ward方法要求欧式距离
        if method == 'ward':
            metric = 'euclidean'

        self._linkage_matrix = linkage(X_std, method=method, metric=metric)
        self._cluster_labels = fcluster(
            self._linkage_matrix, t=self.n_clusters, criterion='maxclust'
        )

        # 将聚类标签加入数据
        self._province_data['cluster'] = self._cluster_labels

        # 计算聚类统计
        self._compute_cluster_stats(features)

        return self._cluster_labels

    def _compute_cluster_stats(self, features):
        """计算各簇统计摘要"""
        stats = {}
        for c in range(1, self.n_clusters + 1):
            mask = self._province_data['cluster'] == c
            members = self._province_data[mask]

            stats[c] = {
                'provinces': list(members.index),
                'n_provinces': len(members),
                'mean_spread_avg': members['mean_spread'].mean(),
                'mean_spread_range': (
                    members['mean_spread'].min(),
                    members['mean_spread'].max()
                ),
                'volatility_avg': members['spread_volatility'].mean(),
                'corr_avg': members['corr_with_national'].mean(),
                'deviation_avg': members['deviation_from_mean'].mean(),
                'risk_level': self._assess_cluster_risk(members)
            }
        self._cluster_stats = stats

    def _assess_cluster_risk(self, members):
        """评估簇内整体风险等级"""
        avg_spread = members['mean_spread'].mean()
        avg_vol = members['spread_volatility'].mean()

        if avg_spread > 90 and avg_vol > 16:
            return '高风险'
        elif avg_spread > 70 and avg_vol > 10:
            return '中等风险'
        elif avg_spread > 55 and avg_vol > 8:
            return '低风险'
        else:
            return '极低风险'

    def get_cluster_stats(self):
        """获取聚类统计摘要"""
        if self._cluster_stats is None:
            self.run_clustering()
        return self._cluster_stats

    # ========================================================================
    # 可视化方法
    # ========================================================================

    def plot_cluster_heatmap(self, theme='light'):
        """
        聚类热力图 - Dendrogram + 热力图联动

        上方: 层次聚类树状图
        下方: 省级利差特征热力图 (按聚类排序)

        参数:
            theme: 'light' 或 'dark'
        """
        if self._cluster_labels is None:
            self.run_clustering()

        config = _get_theme_config(theme)
        features = [
            'mean_spread', 'spread_volatility', 'skewness',
            'corr_with_national', 'deviation_from_mean'
        ]

        # 按聚类排序
        sorted_data = self._province_data.sort_values('cluster')
        X_std, _ = self.get_feature_matrix()
        # 按聚类排序的标准化矩阵
        sorted_idx = sorted_data.index
        X_sorted = self._province_data.loc[sorted_idx, features].values
        X_std_sorted = (X_sorted - X_sorted.mean(axis=0)) / X_sorted.std(axis=0)

        # 特征显示名
        feature_names = ['平均利差', '波动率', '偏度', '国债相关性', '偏离度']

        # 颜色映射
        colorscale = 'RdBu_r' if theme == 'light' else 'RdYlBu_r'

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.25, 0.75],
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('聚类树状图', '省级利差特征热力图')
        )

        # 热力图
        heatmap = go.Heatmap(
            z=X_std_sorted,
            x=feature_names,
            y=list(sorted_idx),
            colorscale=colorscale,
            zmid=0,
            text=[[f'{v:.2f}' for v in row] for row in X_std_sorted],
            texttemplate='%{text}',
            textfont={'size': 9},
            hovertemplate='%{y}<br>%{x}: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title='标准化值',
                thickness=15,
                len=0.7
            ),
            showscale=True
        )
        fig.add_trace(heatmap, row=2, col=1)

        # 聚类分割线
        cluster_boundaries = []
        prev_cluster = None
        for i, province in enumerate(sorted_idx):
            c = sorted_data.loc[province, 'cluster']
            if c != prev_cluster and prev_cluster is not None:
                cluster_boundaries.append(i - 0.5)
            prev_cluster = c

        # 聚类着色带 (作为背景)
        cluster_colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444']
        for i in range(len(cluster_boundaries) + 1):
            start = 0 if i == 0 else cluster_boundaries[i - 1]
            end = cluster_boundaries[i] if i < len(cluster_boundaries) else len(sorted_idx) - 0.5
            cluster_num = sorted_data.iloc[int(start) if start == 0 else int(start + 0.5), -1]
            fig.add_hrect(
                y0=start, y1=end,
                fillcolor=cluster_colors[(cluster_num - 1) % len(cluster_colors)],
                opacity=0.1,
                line_width=0,
                row=2, col=1
            )

        fig.update_layout(
            title='省级利差聚类分析 - 特征热力图',
            height=800,
            template=config['template'],
            paper_bgcolor=config['paper_bgcolor'],
            plot_bgcolor=config['plot_bgcolor'],
            font=dict(color=config['font_color'], size=10),
            margin=dict(l=100, r=30, t=80, b=50)
        )

        fig.update_yaxes(
            autorange='reversed',
            row=2, col=1,
            tickfont=dict(size=9)
        )

        return fig

    def plot_choropleth_map(self, theme='light'):
        """
        省级利差地理分布图 (Choropleth)

        使用省级利差均值进行地理可视化。
        当GeoJSON数据不可用时，自动降级为气泡地图。

        参数:
            theme: 'light' 或 'dark'
        """
        config = _get_theme_config(theme)

        # 准备数据
        data = self._province_data.copy()

        # 省级中心坐标 (用于气泡地图)
        province_coords = _get_province_coordinates()

        # 匹配省份坐标
        valid_provinces = [p for p in data.index if p in province_coords]
        lons = [province_coords[p]['lon'] for p in valid_provinces]
        lats = [province_coords[p]['lat'] for p in valid_provinces]
        spreads = [data.loc[p, 'mean_spread'] for p in valid_provinces]
        sizes = [max(8, data.loc[p, 'mean_spread'] / 5) for p in valid_provinces]

        # 聚类信息 (仅在有聚类结果时)
        if self._cluster_stats and 'cluster' in data.columns:
            hover_texts = [
                f'{p}<br>利差: {data.loc[p, "mean_spread"]:.1f} bps'
                f'<br>簇: {int(data.loc[p, "cluster"])}'
                f'<br>风险: {self._cluster_stats[int(data.loc[p, "cluster"])]["risk_level"]}'
                for p in valid_provinces
            ]
        else:
            hover_texts = [
                f'{p}<br>利差: {data.loc[p, "mean_spread"]:.1f} bps'
                for p in valid_provinces
            ]

        fig = go.Figure()

        # 气泡地图 (可靠方案，不需要GeoJSON)
        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            text=hover_texts,
            marker=dict(
                size=sizes,
                color=spreads,
                colorscale='YlOrRd',
                reversescale=False,
                cmin=data['mean_spread'].min(),
                cmax=data['mean_spread'].max(),
                colorbar=dict(title='利差 (bps)'),
                line=dict(width=1, color='white'),
                opacity=0.85,
                sizemode='diameter'
            ),
            hovertemplate='%{text}<extra></extra>',
            name='省级利差'
        ))

        fig.update_geos(
            scope='asia',
            center=dict(lat=35, lon=105),
            projection_scale=5,
            showland=True,
            landcolor='#e8e8e8' if theme == 'light' else '#2d2d2d',
            showocean=True,
            oceancolor='#cfe2f3' if theme == 'light' else '#1a3a5f',
            showlakes=True,
            lakecolor='#cfe2f3' if theme == 'light' else '#1a3a5f',
            showcountries=True,
            countrycolor='#999999',
            showsubunits=True,
            subunitcolor='#cccccc' if theme == 'light' else '#555555',
            fitbounds='locations'
        )

        fig.update_layout(
            title='省级地方债利差地理分布',
            height=600,
            template=config['template'],
            paper_bgcolor=config['paper_bgcolor'],
            geo=dict(
                bgcolor=config['plot_bgcolor']
            ),
            font=dict(color=config['font_color']),
            margin=dict(l=10, r=10, t=60, b=20)
        )

        return fig

    def plot_cluster_comparison(self, theme='light'):
        """
        聚类对比雷达图 - 各簇特征对比

        展示4个聚类在5个维度上的特征均值对比。

        参数:
            theme: 'light' 或 'dark'
        """
        if self._cluster_stats is None:
            self.run_clustering()

        config = _get_theme_config(theme)
        features = [
            'mean_spread', 'spread_volatility', 'skewness',
            'corr_with_national', 'deviation_from_mean'
        ]
        feature_names = ['平均利差', '波动率', '偏度', '国债相关性', '偏离度']

        # 标准化各簇均值用于雷达图
        X = self._province_data[features].values
        means = X.mean(axis=0)
        stds = X.std(axis=0)

        cluster_colors = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444']
        cluster_names = {
            1: '簇1', 2: '簇2', 3: '簇3', 4: '簇4'
        }

        fig = go.Figure()

        for c in range(1, self.n_clusters + 1):
            mask = self._province_data['cluster'] == c
            cluster_mean = self._province_data.loc[mask, features].mean().values
            # 标准化到0-1范围
            normalized = (cluster_mean - means) / stds
            normalized = np.clip(normalized, -3, 3) / 3  # 映射到[-1,1]
            normalized = (normalized + 1) / 2  # 映射到[0,1]

            risk = self._cluster_stats[c]['risk_level']
            n_prov = self._cluster_stats[c]['n_provinces']

            fig.add_trace(go.Scatterpolar(
                r=normalized,
                theta=feature_names,
                fill='toself',
                name=f'{cluster_names.get(c, f"簇{c}")} ({risk}, {n_prov}省)',
                line=dict(color=cluster_colors[(c - 1) % len(cluster_colors)], width=2),
                fillcolor=cluster_colors[(c - 1) % len(cluster_colors)],
                opacity=0.25
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(color=config['font_color'], size=9),
                    gridcolor=config['grid_color']
                ),
                angularaxis=dict(
                    tickfont=dict(color=config['font_color'], size=11),
                    gridcolor=config['grid_color']
                ),
                bgcolor=config['plot_bgcolor']
            ),
            title='省级聚类特征对比雷达图',
            height=500,
            template=config['template'],
            paper_bgcolor=config['paper_bgcolor'],
            font=dict(color=config['font_color']),
            legend=dict(
                orientation='h',
                yanchor='bottom', y=-0.2,
                xanchor='center', x=0.5,
                font=dict(size=10)
            )
        )

        return fig


# ============================================================================
# 辅助函数
# ============================================================================

def _get_theme_config(theme='light'):
    """获取主题配置"""
    if theme == 'dark':
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font_color': '#F8FAFC',
            'grid_color': 'rgba(71, 85, 105, 0.3)',
        }
    return {
        'template': 'none',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font_color': '#0F172A',
        'grid_color': 'rgba(226, 232, 240, 0.8)',
    }


def _get_province_coordinates():
    """获取各省中心地理坐标 (用于气泡地图)"""
    return {
        '北京': {'lat': 39.9, 'lon': 116.4},
        '天津': {'lat': 39.1, 'lon': 117.2},
        '河北': {'lat': 38.0, 'lon': 114.5},
        '山西': {'lat': 37.6, 'lon': 112.3},
        '内蒙古': {'lat': 40.8, 'lon': 111.7},
        '辽宁': {'lat': 41.8, 'lon': 123.4},
        '吉林': {'lat': 43.9, 'lon': 125.3},
        '黑龙江': {'lat': 45.8, 'lon': 126.6},
        '上海': {'lat': 31.2, 'lon': 121.5},
        '江苏': {'lat': 32.1, 'lon': 118.8},
        '浙江': {'lat': 29.1, 'lon': 120.2},
        '安徽': {'lat': 31.9, 'lon': 117.3},
        '福建': {'lat': 26.1, 'lon': 119.3},
        '江西': {'lat': 27.6, 'lon': 115.9},
        '山东': {'lat': 36.7, 'lon': 117.0},
        '河南': {'lat': 34.8, 'lon': 113.7},
        '湖北': {'lat': 31.0, 'lon': 112.3},
        '湖南': {'lat': 27.6, 'lon': 112.0},
        '广东': {'lat': 23.1, 'lon': 113.3},
        '广西': {'lat': 22.8, 'lon': 108.3},
        '海南': {'lat': 19.2, 'lon': 109.7},
        '重庆': {'lat': 29.6, 'lon': 106.5},
        '四川': {'lat': 30.6, 'lon': 104.1},
        '贵州': {'lat': 26.6, 'lon': 106.7},
        '云南': {'lat': 25.0, 'lon': 102.7},
        '西藏': {'lat': 29.7, 'lon': 91.1},
        '陕西': {'lat': 34.3, 'lon': 108.9},
        '甘肃': {'lat': 36.1, 'lon': 103.8},
        '青海': {'lat': 36.6, 'lon': 101.8},
        '宁夏': {'lat': 37.3, 'lon': 106.2},
        '新疆': {'lat': 43.8, 'lon': 87.6}
    }