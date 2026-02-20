"""Тесты для UniverseAnalyzer и MultiverseDynamicsExplorer."""

import pytest
from multiverse_tester.core import (
    UniverseParameters,
    UniverseAnalyzer,
    MultiverseDynamicsExplorer,
    HabitabilityIndex,
)


# ==================== UniverseAnalyzer ====================


class TestUniverseAnalyzer:
    def test_calculate_habitability_index_returns_tuple(self):
        u = UniverseParameters()
        analyzer = UniverseAnalyzer(u)
        index, score, metrics = analyzer.calculate_habitability_index()
        assert isinstance(index, HabitabilityIndex)
        assert isinstance(score, (int, float))
        assert isinstance(metrics, dict)

    def test_our_universe_is_habitable_or_better(self):
        u = UniverseParameters(name="Наша Вселенная")
        analyzer = UniverseAnalyzer(u)
        index, score, metrics = analyzer.calculate_habitability_index()
        assert score > 0.3
        assert index in (HabitabilityIndex.HABITABLE, HabitabilityIndex.OPTIMAL)

    def test_metrics_keys(self):
        u = UniverseParameters()
        analyzer = UniverseAnalyzer(u)
        _, _, metrics = analyzer.calculate_habitability_index()
        expected = ['atomic', 'chemistry', 'nuclear', 'carbon', 'heavy_elements',
                    'fusion', 'supernova', 'r_process', 'quantum_scale',
                    'cosmology_H', 'cosmology_Lambda']
        for key in expected:
            assert key in metrics
            assert metrics[key] in (0.0, 0.5, 1.0)

    def test_score_in_range(self):
        u = UniverseParameters()
        analyzer = UniverseAnalyzer(u)
        _, score, _ = analyzer.calculate_habitability_index()
        assert 0 <= score <= 1

    def test_extreme_alpha_dead_universe(self):
        u = UniverseParameters(alpha=0.5)
        analyzer = UniverseAnalyzer(u)
        index, score, _ = analyzer.calculate_habitability_index()
        assert score < 0.5
        assert index in (HabitabilityIndex.DEAD, HabitabilityIndex.HOSTILE)

    def test_get_all_properties(self):
        u = UniverseParameters()
        analyzer = UniverseAnalyzer(u)
        props = analyzer.get_all_properties()
        assert 'alpha' in props
        assert 'e_ratio' in props
        assert 'bohr_radius_norm' in props
        assert 'binding_energy_mev' in props
        assert 'pp_rate' in props
        assert 'triple_alpha_rate' in props
        assert 'carbon_prod' in props


# ==================== HabitabilityIndex ====================


class TestHabitabilityIndex:
    def test_enum_values(self):
        assert HabitabilityIndex.DEAD.value == 0
        assert HabitabilityIndex.HOSTILE.value == 1
        assert HabitabilityIndex.MARGINAL.value == 2
        assert HabitabilityIndex.HABITABLE.value == 3
        assert HabitabilityIndex.OPTIMAL.value == 4


# ==================== MultiverseDynamicsExplorer ====================


class TestMultiverseDynamicsExplorer:
    def test_scan_parameter_alpha(self):
        explorer = MultiverseDynamicsExplorer()
        result = explorer.scan_parameter(
            param_name="alpha",
            start=1/500,
            stop=1/50,
            num_points=10,
        )
        assert result['param_name'] == "alpha"
        assert len(result['param_values']) == 10
        assert len(result['habitability_scores']) == 10
        assert len(result['properties']) == 10

    def test_scan_parameter_m_p(self):
        explorer = MultiverseDynamicsExplorer()
        result = explorer.scan_parameter(
            param_name="m_p",
            start=0.5,
            stop=2.0,
            num_points=5,
        )
        assert result['param_name'] == "m_p"
        assert len(result['param_values']) == 5

    def test_analyze_correlations_after_scan(self):
        explorer = MultiverseDynamicsExplorer()
        explorer.scan_parameter("alpha", 1/300, 1/50, num_points=20)
        corr = explorer.analyze_correlations("alpha")
        assert isinstance(corr, dict)
        assert len(corr) > 0

    def test_analyze_correlations_raises_without_scan(self):
        explorer = MultiverseDynamicsExplorer()
        with pytest.raises(ValueError, match="Сначала выполните сканирование"):
            explorer.analyze_correlations("alpha")

    def test_scan_parameter_unknown_raises(self):
        explorer = MultiverseDynamicsExplorer()
        with pytest.raises(ValueError, match="Unknown parameter"):
            explorer.scan_parameter("unknown_param", 0, 1, num_points=5)
