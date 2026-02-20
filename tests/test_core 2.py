"""Tests for multiverse_tester.core module."""

import math
import pytest
from multiverse_tester import (
    UniversalConstants,
    UniverseParameters,
    UniverseAnalyzer,
    AtomicPhysics,
    NuclearPhysics,
    StellarNucleosynthesis,
    HabitabilityIndex,
)


class TestUniversalConstants:
    """Tests for UniversalConstants."""

    def test_default_values(self):
        const = UniversalConstants()
        assert const.c == 299792458.0
        assert const.m_p > 0
        assert const.m_e > 0
        assert const.G > 0
        assert const.hbar > 0
        assert const.epsilon_0 > 0


class TestUniverseParameters:
    """Tests for UniverseParameters."""

    def test_default_universe(self):
        u = UniverseParameters()
        assert u.alpha == pytest.approx(1/137.036, rel=1e-3)
        assert u.m_p == pytest.approx(UniversalConstants().m_p)
        assert u.e == pytest.approx(UniversalConstants().e)

    def test_custom_alpha(self):
        alpha_val = 1/100
        u = UniverseParameters(alpha=alpha_val)
        assert u.alpha == alpha_val

    def test_custom_m_p(self):
        const = UniversalConstants()
        u = UniverseParameters(m_p=2 * const.m_p)
        assert u.m_p == pytest.approx(2 * const.m_p)

    def test_planck_units(self):
        u = UniverseParameters()
        assert u.m_planck > 0
        assert u.l_planck > 0
        assert u.t_planck > 0

    def test_nuclear_masses_scale_with_m_p(self):
        const = UniversalConstants()
        u1 = UniverseParameters(m_p=const.m_p)
        u2 = UniverseParameters(m_p=2 * const.m_p)
        assert u2.m_he4 == pytest.approx(2 * u1.m_he4)

    def test_repr(self):
        u = UniverseParameters(name="Test")
        r = repr(u)
        assert "Test" in r
        assert "α=" in r


class TestAtomicPhysics:
    """Tests for AtomicPhysics."""

    def test_bohr_radius_our_universe(self):
        u = UniverseParameters()
        atom = AtomicPhysics(u)
        a0 = atom.bohr_radius()
        assert a0 == pytest.approx(5.29e-11, rel=0.01)

    def test_rydberg_ev_our_universe(self):
        u = UniverseParameters()
        atom = AtomicPhysics(u)
        E = atom.rydberg_ev()
        assert E == pytest.approx(13.6, rel=0.01)

    def test_fine_structure_effects(self):
        u = UniverseParameters()
        atom = AtomicPhysics(u)
        effects = atom.fine_structure_effects()
        assert 'a0' in effects
        assert 'a0_norm' in effects
        assert 'a0_over_λc' in effects
        assert effects['a0_norm'] == pytest.approx(1.0, rel=0.02)


class TestNuclearPhysics:
    """Tests for NuclearPhysics."""

    def test_binding_energy_fe56(self):
        u = UniverseParameters()
        nuc = NuclearPhysics(u)
        B_per_nucleon = nuc.binding_per_nucleon(56, 26)
        assert 7 < B_per_nucleon < 10

    def test_coulomb_barrier(self):
        u = UniverseParameters()
        nuc = NuclearPhysics(u)
        barrier = nuc.coulomb_barrier(1, 1, 1, 1)
        assert barrier > 0


class TestHabitabilityIndex:
    """Tests for HabitabilityIndex enum."""

    def test_values(self):
        assert HabitabilityIndex.DEAD.value == 0
        assert HabitabilityIndex.OPTIMAL.value == 4

    def test_names(self):
        assert HabitabilityIndex.HABITABLE.name == "HABITABLE"


@pytest.fixture
def our_universe():
    return UniverseParameters()


@pytest.fixture
def our_analyzer(our_universe):
    return UniverseAnalyzer(our_universe)


class TestUniverseAnalyzer:
    """Tests for UniverseAnalyzer."""

    def test_our_universe_habitable(self, our_universe, our_analyzer):
        index, score, metrics = our_analyzer.calculate_habitability_index()
        assert 0 <= score <= 1
        assert index in HabitabilityIndex
        assert len(metrics) > 0
        # Our universe should be at least marginal
        assert score > 0.3

    def test_dead_universe(self):
        u = UniverseParameters(alpha=1/10)  # Extreme alpha
        analyzer = UniverseAnalyzer(u)
        index, score, metrics = analyzer.calculate_habitability_index()
        assert score < 0.6

    def test_get_all_properties(self, our_analyzer):
        props = our_analyzer.get_all_properties()
        assert 'alpha' in props
        assert 'pp_rate' in props
        assert 'triple_alpha_rate' in props
        assert 'binding_energy_mev' in props
