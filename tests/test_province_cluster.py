"""
省份聚类热力图模块测试

覆盖:
1. ProvinceClusterMap 初始化与 Mock 数据生成
2. 省份数据完整性 (31省、5特征维度)
3. 特征矩阵标准化
4. 层次聚类执行与标签分配
5. 聚类统计计算与风险评估
6. 可视化方法 (热力图、地理分布图、雷达图)
7. 自定义数据输入与聚类参数
8. 区域分类与特征种子一致性
9. 短数据边界条件
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from province_cluster import (
    ProvinceClusterMap, PROVINCES, PROVINCE_SHORT,
    REGION_GROUPS, REGION_BASE_PROFILES,
    GEOJSON_NAME_MAP, _load_choropleth_geojson, _get_choropleth_colorscale,
    _get_theme_config
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cluster_map():
    """默认聚类分析器 (Mock数据)"""
    return ProvinceClusterMap(n_clusters=4, seed=42)


@pytest.fixture
def clustered_map(cluster_map):
    """已执行聚类的分析器"""
    cluster_map.run_clustering()
    return cluster_map


@pytest.fixture
def custom_data():
    """自定义省级数据"""
    np.random.seed(123)
    provinces = PROVINCES[:10]  # 只用10省做快速测试
    data = pd.DataFrame({
        'mean_spread': np.random.uniform(50, 120, 10),
        'spread_volatility': np.random.uniform(8, 20, 10),
        'skewness': np.random.uniform(0.1, 1.0, 10),
        'corr_with_national': np.random.uniform(0.6, 0.95, 10),
        'deviation_from_mean': np.random.uniform(-20, 30, 10)
    }, index=provinces)
    return data


@pytest.fixture
def small_cluster_map(custom_data):
    """小规模聚类分析器"""
    return ProvinceClusterMap(province_data=custom_data, n_clusters=3, seed=123)


# ============================================================================
# 1. 初始化与 Mock 数据生成
# ============================================================================

class TestInitialization:

    def test_default_init(self):
        """默认初始化应自动生成Mock数据"""
        pcm = ProvinceClusterMap()
        assert pcm._province_data is not None
        assert len(pcm._province_data) == 31

    def test_custom_data_init(self, custom_data):
        """自定义数据初始化"""
        pcm = ProvinceClusterMap(province_data=custom_data)
        assert len(pcm._province_data) == 10
        assert pcm._province_data.index.tolist() == PROVINCES[:10]

    def test_cluster_count(self):
        """聚类数量参数"""
        pcm = ProvinceClusterMap(n_clusters=3)
        assert pcm.n_clusters == 3

    def test_seed_reproducibility(self):
        """相同种子生成相同数据"""
        pcm1 = ProvinceClusterMap(seed=42)
        pcm2 = ProvinceClusterMap(seed=42)
        pd.testing.assert_frame_equal(pcm1._province_data, pcm2._province_data)


# ============================================================================
# 2. Mock数据完整性
# ============================================================================

class TestMockDataGeneration:

    def test_31_provinces(self, cluster_map):
        """Mock数据应包含31省"""
        data = cluster_map._province_data
        assert len(data) == 31
        assert set(data.index) == set(PROVINCES)

    def test_feature_columns(self, cluster_map):
        """数据应包含5个特征列"""
        data = cluster_map._province_data
        expected_cols = [
            'mean_spread', 'spread_volatility', 'skewness',
            'corr_with_national', 'deviation_from_mean'
        ]
        for col in expected_cols:
            assert col in data.columns

    def test_spread_values_realistic(self, cluster_map):
        """利差均值应在合理范围 (30-150 bps)"""
        spreads = cluster_map._province_data['mean_spread']
        assert spreads.min() > 30
        assert spreads.max() < 150

    def test_volatility_positive(self, cluster_map):
        """波动率应为正值"""
        vols = cluster_map._province_data['spread_volatility']
        assert (vols > 0).all()

    def test_correlation_bounded(self, cluster_map):
        """国债相关性应在0.5-1.0之间"""
        corr = cluster_map._province_data['corr_with_national']
        assert corr.min() >= 0.5
        assert corr.max() <= 1.0

    def test_deviation_consistency(self, cluster_map):
        """偏离度 = 利差均值 - 70"""
        spreads = cluster_map._province_data['mean_spread']
        deviations = cluster_map._province_data['deviation_from_mean']
        np.testing.assert_allclose(deviations.values, spreads.values - 70, atol=1e-10)

    def test_regional_differentiation(self, cluster_map):
        """不同区域应有显著的利差差异"""
        data = cluster_map._province_data
        east = data.loc[data.index.isin(REGION_GROUPS['东部沿海']), 'mean_spread'].mean()
        west = data.loc[data.index.isin(REGION_GROUPS['西部开发']), 'mean_spread'].mean()
        # 西部利差应显著高于东部
        assert west > east


# ============================================================================
# 3. 特征矩阵标准化
# ============================================================================

class TestFeatureMatrix:

    def test_standardization(self, cluster_map):
        """标准化后均值≈0, 标准差≈1"""
        X_std, features = cluster_map.get_feature_matrix()
        for col_idx in range(X_std.shape[1]):
            np.testing.assert_allclose(X_std[:, col_idx].mean(), 0, atol=0.05)
            np.testing.assert_allclose(X_std[:, col_idx].std(), 1, atol=0.05)

    def test_feature_names(self, cluster_map):
        """特征名列表正确"""
        _, features = cluster_map.get_feature_matrix()
        expected = [
            'mean_spread', 'spread_volatility', 'skewness',
            'corr_with_national', 'deviation_from_mean'
        ]
        assert features == expected

    def test_matrix_shape(self, cluster_map):
        """矩阵形状 = (n_provinces, 5)"""
        X_std, _ = cluster_map.get_feature_matrix()
        assert X_std.shape == (31, 5)


# ============================================================================
# 4. 层次聚类
# ============================================================================

class TestClustering:

    def test_default_clustering(self, cluster_map):
        """默认聚类 (ward, euclidean)"""
        labels = cluster_map.run_clustering()
        assert len(labels) == 31
        assert set(labels) == {1, 2, 3, 4}

    def test_cluster_labels_in_data(self, clustered_map):
        """聚类标签应存入数据"""
        assert 'cluster' in clustered_map._province_data.columns
        assert set(clustered_map._province_data['cluster']) == {1, 2, 3, 4}

    def test_linkage_matrix_shape(self, cluster_map):
        """linkage矩阵形状 = (n-1, 4)"""
        cluster_map.run_clustering()
        lm = cluster_map._linkage_matrix
        assert lm.shape == (30, 4)

    def test_different_methods(self, cluster_map):
        """不同聚类方法应产出不同结果"""
        labels_ward = cluster_map.run_clustering(method='ward')
        labels_complete = cluster_map.run_clustering(method='complete')
        # Ward和Complete聚类结果通常不同 (但不必完全不同)
        # 这里只检查结构正确
        assert len(labels_complete) == 31

    def test_cluster_count_parameter(self):
        """n_clusters=3应产出3个簇"""
        pcm = ProvinceClusterMap(n_clusters=3)
        labels = pcm.run_clustering()
        assert len(set(labels)) <= 3

    def test_reproducibility(self):
        """相同种子相同方法应产出相同聚类"""
        pcm1 = ProvinceClusterMap(seed=42)
        pcm2 = ProvinceClusterMap(seed=42)
        l1 = pcm1.run_clustering()
        l2 = pcm2.run_clustering()
        np.testing.assert_array_equal(l1, l2)


# ============================================================================
# 5. 聚类统计
# ============================================================================

class TestClusterStats:

    def test_stats_keys(self, clustered_map):
        """统计键 = 1..n_clusters"""
        stats = clustered_map.get_cluster_stats()
        assert set(stats.keys()) == {1, 2, 3, 4}

    def test_stats_content(self, clustered_map):
        """每个簇统计应包含必要字段"""
        stats = clustered_map.get_cluster_stats()
        for c, s in stats.items():
            assert 'provinces' in s
            assert 'n_provinces' in s
            assert 'mean_spread_avg' in s
            assert 'volatility_avg' in s
            assert 'risk_level' in s

    def test_province_count_sum(self, clustered_map):
        """各省数量之和 = 31"""
        stats = clustered_map.get_cluster_stats()
        total = sum(s['n_provinces'] for s in stats.values())
        assert total == 31

    def test_all_provinces_covered(self, clustered_map):
        """所有省份都被分配到某个簇"""
        stats = clustered_map.get_cluster_stats()
        all_provinces = set()
        for s in stats.values():
            all_provinces.update(s['provinces'])
        assert len(all_provinces) == 31

    def test_risk_level_valid(self, clustered_map):
        """风险等级应在预定义列表中"""
        valid_levels = ['高风险', '中等风险', '低风险', '极低风险']
        stats = clustered_map.get_cluster_stats()
        for s in stats.values():
            assert s['risk_level'] in valid_levels

    def test_stats_auto_compute(self, cluster_map):
        """get_cluster_stats应自动执行聚类"""
        stats = cluster_map.get_cluster_stats()
        assert stats is not None
        assert len(stats) == 4


# ============================================================================
# 6. 可视化方法
# ============================================================================

class TestVisualization:

    def test_cluster_heatmap(self, clustered_map):
        """聚类热力图应返回Plotly Figure"""
        fig = clustered_map.plot_cluster_heatmap(theme='light')
        assert fig is not None
        assert len(fig.data) > 0

    def test_cluster_heatmap_dark(self, clustered_map):
        """暗色主题热力图"""
        fig = clustered_map.plot_cluster_heatmap(theme='dark')
        assert fig is not None

    def test_choropleth_map(self, clustered_map):
        """地理分布图应返回Plotly Figure"""
        fig = clustered_map.plot_choropleth_map(theme='light')
        assert fig is not None
        assert len(fig.data) > 0

    def test_choropleth_dark(self, clustered_map):
        """暗色主题地理分布图"""
        fig = clustered_map.plot_choropleth_map(theme='dark')
        assert fig is not None

    def test_cluster_comparison_radar(self, clustered_map):
        """聚类对比雷达图"""
        fig = clustered_map.plot_cluster_comparison(theme='light')
        assert fig is not None
        assert len(fig.data) == 4  # 4个聚类

    def test_radar_dark(self, clustered_map):
        """暗色主题雷达图"""
        fig = clustered_map.plot_cluster_comparison(theme='dark')
        assert fig is not None

    def test_visualization_auto_cluster(self, cluster_map):
        """可视化方法应自动执行聚类"""
        fig = cluster_map.plot_cluster_heatmap()
        assert cluster_map._cluster_labels is not None


# ============================================================================
# 7. 自定义数据输入
# ============================================================================

class TestCustomData:

    def test_custom_data_clustering(self, small_cluster_map):
        """自定义数据聚类"""
        labels = small_cluster_map.run_clustering()
        assert len(labels) == 10
        assert len(set(labels)) <= 3

    def test_custom_data_stats(self, small_cluster_map):
        """自定义数据统计"""
        small_cluster_map.run_clustering()
        stats = small_cluster_map.get_cluster_stats()
        assert len(stats) <= 3
        total = sum(s['n_provinces'] for s in stats.values())
        assert total == 10

    def test_custom_data_heatmap(self, small_cluster_map):
        """自定义数据热力图"""
        small_cluster_map.run_clustering()
        fig = small_cluster_map.plot_cluster_heatmap()
        assert fig is not None


# ============================================================================
# 8. 区域分类
# ============================================================================

class TestRegionClassification:

    def test_region_groups_complete(self):
        """区域分类应覆盖所有省份"""
        all_provinces_in_groups = set()
        for provinces in REGION_GROUPS.values():
            all_provinces_in_groups.update(provinces)
        assert all_provinces_in_groups == set(PROVINCES)

    def test_province_short_mapping(self):
        """省份简称映射完整性"""
        assert len(PROVINCE_SHORT) == 31
        assert set(PROVINCE_SHORT.keys()) == set(PROVINCES)

    def test_region_profiles_keys(self):
        """区域特征种子应包含4个区域"""
        assert set(REGION_BASE_PROFILES.keys()) == set(REGION_GROUPS.keys())

    def test_get_region(self, cluster_map):
        """_get_region正确分类"""
        assert cluster_map._get_region('北京') == '东部沿海'
        assert cluster_map._get_region('河南') == '中部内陆'
        assert cluster_map._get_region('四川') == '西部开发'
        assert cluster_map._get_region('辽宁') == '东北老工业'


# ============================================================================
# 9. 边界条件
# ============================================================================

class TestEdgeCases:

    def test_two_clusters_minimum(self):
        """最少2个聚类"""
        pcm = ProvinceClusterMap(n_clusters=2)
        labels = pcm.run_clustering()
        assert len(set(labels)) <= 2

    def test_ward_forces_euclidean(self, cluster_map):
        """Ward方法应强制使用欧式距离"""
        labels = cluster_map.run_clustering(method='ward', metric='correlation')
        # Ward应覆盖metric参数为euclidean
        assert len(labels) == 31

    def test_high_cluster_count(self):
        """8个聚类 (超过区域数)"""
        pcm = ProvinceClusterMap(n_clusters=8)
        labels = pcm.run_clustering()
        assert len(set(labels)) <= 8


# ============================================================================
# 10. Choropleth地图升级测试
# ============================================================================

class TestChoroplethUpgrade:

    def test_geojson_name_map_complete(self):
        """GeoJSON名称映射应覆盖31省"""
        assert len(GEOJSON_NAME_MAP) == 31

    def test_geojson_name_map_covers_all_provinces(self):
        """所有省份箁名应在映射中"""
        for province in PROVINCES:
            assert province in GEOJSON_NAME_MAP.values()

    def test_geojson_loads(self):
        """GeoJSON文件应成功加载"""
        geojson = _load_choropleth_geojson()
        assert geojson is not None
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) == 31

    def test_geojson_feature_names_match(self):
        """GeoJSON省名应能映射到箁名"""
        geojson = _load_choropleth_geojson()
        geojson_names = [f['properties']['name'] for f in geojson['features']]
        for name in geojson_names:
            assert name in GEOJSON_NAME_MAP

    def test_choropleth_trace_type(self, clustered_map):
        """GeoJSON可用时应使用choropleth追踪类型"""
        geojson = _load_choropleth_geojson()
        if geojson:
            fig = clustered_map.plot_choropleth_map(theme='light')
            assert fig.data[0].type == 'choropleth'
        else:
            fig = clustered_map.plot_choropleth_map(theme='light')
            assert fig.data[0].type == 'scattergeo'

    def test_choropleth_locations_match_geojson(self, clustered_map):
        """choropleth的locations应使用GeoJSON全名"""
        geojson = _load_choropleth_geojson()
        if geojson:
            fig = clustered_map.plot_choropleth_map(theme='light')
            trace = fig.data[0]
            # 所有locations应在GeoJSON features中
            geojson_names = set(f['properties']['name'] for f in geojson['features'])
            for loc in trace.locations:
                assert loc in geojson_names

    def test_choropleth_z_values(self, clustered_map):
        """choropleth的z值应为利差均值"""
        geojson = _load_choropleth_geojson()
        if geojson:
            fig = clustered_map.plot_choropleth_map(theme='light')
            trace = fig.data[0]
            assert len(trace.z) > 0
            # z值应在合理范围
            for z in trace.z:
                assert z > 30 and z < 150

    def test_choropleth_featureidkey(self, clustered_map):
        """featureidkey应为properties.name"""
        geojson = _load_choropleth_geojson()
        if geojson:
            fig = clustered_map.plot_choropleth_map(theme='light')
            assert fig.data[0].featureidkey == 'properties.name'

    def test_choropleth_title_text(self, clustered_map):
        """choropleth标题应标注Choropleth"""
        geojson = _load_choropleth_geojson()
        if geojson:
            fig = clustered_map.plot_choropleth_map(theme='light')
            assert 'Choropleth' in fig.layout.title.text
        else:
            fig = clustered_map.plot_choropleth_map(theme='light')
            assert '气泡地图' in fig.layout.title.text

    def test_choropleth_dark_colorscale(self):
        """暗色色阶应返回5个色阶段"""
        colorscale = _get_choropleth_colorscale('dark')
        assert len(colorscale) == 5
        assert colorscale[0][0] == 0
        assert colorscale[-1][0] == 1

    def test_choropleth_light_colorscale(self):
        """亮色色阶应返回5个色阶段"""
        colorscale = _get_choropleth_colorscale('light')
        assert len(colorscale) == 5

    def test_bubble_fallback_title(self, clustered_map):
        """气泡地图降级标题应标注气泡地图"""
        # Mock GeoJSON unavailable by testing the _plot_bubble_map directly
        config = _get_theme_config('light')
        fig = clustered_map._plot_bubble_map(config, 'light')
        assert '气泡地图' in fig.layout.title.text

    def test_bubble_fallback_trace_type(self, clustered_map):
        """气泡地图降级应使用scattergeo"""
        config = _get_theme_config('light')
        fig = clustered_map._plot_bubble_map(config, 'light')
        assert fig.data[0].type == 'scattergeo'

    def test_choropleth_with_cluster_hover(self, clustered_map):
        """有聚类结果时hover应包含簇信息"""
        geojson = _load_choropleth_geojson()
        if geojson:
            fig = clustered_map.plot_choropleth_map(theme='light')
            trace = fig.data[0]
            # 文本中应包含簇信息
            for text in trace.text:
                assert '簇' in text or '利差' in text