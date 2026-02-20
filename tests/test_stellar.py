"""Тесты для StellarNucleosynthesis."""

import pytest
from multiverse_tester.core import (
    UniverseParameters,
    NuclearPhysics,
    StellarNucleosynthesis,
)


class TestStellarNucleosynthesis:
    @pytest.fixture
    def stellar(self):
        u = UniverseParameters()
        nuclear = NuclearPhysics(u)
        return StellarNucleosynthesis(u, nuclear)

    def test_pp_chain_rate_returns_dict(self, stellar):
        result = stellar.pp_chain_rate()
        assert 'rate_relative' in result
        assert 'gamow_factor' in result
        assert 'tau_hydrogen_years' in result
        assert 'barrier_mev' in result

    def test_pp_chain_rate_our_universe_near_one(self, stellar):
        result = stellar.pp_chain_rate()
        assert 0.5 < result['rate_relative'] < 2.0

    def test_cno_cycle_rate(self, stellar):
        result = stellar.cno_cycle_rate()
        assert 'rate_relative' in result
        assert 'gamow_factor' in result
        assert result['gamow_factor'] > 0

    def test_triple_alpha_returns_dict(self, stellar):
        result = stellar.triple_alpha()
        assert 'rate_relative' in result
        assert 'resonance_factor' in result
        assert 'carbon_production' in result

    def test_carbon_burning(self, stellar):
        result = stellar.carbon_burning()
        assert 'rate_relative' in result
        assert 'T_ignition' in result
        assert result['T_ignition'] > 0

    def test_alpha_process_returns_list(self, stellar):
        results = stellar.alpha_process()
        assert isinstance(results, list)
        assert len(results) > 0
        assert 'nucleus' in results[0]
        assert 'relative_yield' in results[0]

    def test_s_process(self, stellar):
        result = stellar.s_process()
        assert 'path' in result
        assert 'capture_time_years' in result
        assert 'beta_lifetime_years' in result

    def test_r_process(self, stellar):
        result = stellar.r_process()
        assert 'max_neutron_excess' in result
        assert 'transuranic_elements' in result
        assert result['transuranic_elements'] >= 0

    def test_supernova_nucleosynthesis(self, stellar):
        result = stellar.supernova_nucleosynthesis(progenitor_mass=15)
        assert 'elements' in result
        assert 'fe_core_mass' in result
        assert 'collapse_type' in result
        assert 'neutron_star_formed' in result
        assert result['collapse_type'] in ["черная дыра", "белый карлик", "нейтронная звезда"]

    def test_complete_nucleosynthesis_analysis(self, stellar):
        result = stellar.complete_nucleosynthesis_analysis()
        assert 'pp_chain' in result
        assert 'cno_cycle' in result
        assert 'triple_alpha' in result
        assert 'carbon_burning' in result
        assert 'alpha_process' in result
        assert 's_process' in result
        assert 'r_process' in result
        assert 'supernova' in result
        assert 'element_production' in result
