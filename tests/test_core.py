"""Тесты для модуля core: UniversalConstants, UniverseParameters, AtomicPhysics, NuclearPhysics."""

import math
import pytest
from multiverse_tester.core import (
    UniversalConstants,
    UniverseParameters,
    AtomicPhysics,
    NuclearPhysics,
    HabitabilityIndex,
)


# ==================== UniversalConstants ====================


class TestUniversalConstants:
    def test_default_values_exist(self):
        const = UniversalConstants()
        assert const.hbar > 0
        assert const.c > 0
        assert const.epsilon_0 > 0
        assert const.G > 0
        assert const.m_e > 0
        assert const.m_p > 0
        assert const.e > 0
        assert const.k_B > 0

    def test_speed_of_light(self):
        const = UniversalConstants()
        assert abs(const.c - 299792458.0) < 1

    def test_fine_structure_alpha_from_constants(self):
        const = UniversalConstants()
        alpha_calc = const.e**2 / (4 * math.pi * const.epsilon_0 * const.hbar * const.c)
        assert 0.007 < alpha_calc < 0.008


# ==================== UniverseParameters ====================


class TestUniverseParameters:
    def test_default_universe(self):
        u = UniverseParameters()
        assert u.name == "Our Universe"
        assert 0.007 < u.alpha < 0.008
        assert u.m_p > 0
        assert u.m_e > 0

    def test_custom_alpha(self):
        u = UniverseParameters(name="Test", alpha=1/137)
        assert abs(u.alpha - 1/137) < 1e-10
        assert u.name == "Test"

    def test_custom_m_p(self):
        u = UniverseParameters(m_p=2e-27)
        assert u.m_p == 2e-27

    def test_planck_units_positive(self):
        u = UniverseParameters()
        assert u.m_planck > 0
        assert u.l_planck > 0
        assert u.t_planck > 0
        assert u.q_planck > 0

    def test_alpha_e_consistency(self):
        u = UniverseParameters(alpha=0.01)
        e_expected = math.sqrt(0.01 * 4 * math.pi * u.epsilon_0 * u.hbar * u.c)
        assert abs(u.e - e_expected) < 1e-30

    def test_fix_e_mode(self):
        e_val = 1.6e-19
        u = UniverseParameters(fix_e=True, e=e_val)
        alpha_from_e = (e_val**2) / (4 * math.pi * u.epsilon_0 * u.hbar * u.c)
        assert abs(u.alpha - alpha_from_e) < 1e-35

    def test_nuclear_masses_scale_with_m_p(self):
        u1 = UniverseParameters(m_p=1.6726219e-27)
        u2 = UniverseParameters(m_p=2 * 1.6726219e-27)
        ratio = u2.m_fe56 / u1.m_fe56
        assert 1.9 < ratio < 2.1


# ==================== AtomicPhysics ====================


class TestAtomicPhysics:
    def test_bohr_radius_default_universe(self):
        u = UniverseParameters()
        atomic = AtomicPhysics(u)
        a0 = atomic.bohr_radius()
        assert 4e-11 < a0 < 6e-11

    def test_rydberg_energy_default(self):
        u = UniverseParameters()
        atomic = AtomicPhysics(u)
        E_ryd = atomic.rydberg_ev()
        assert 12 < E_ryd < 15

    def test_compton_wavelength_positive(self):
        u = UniverseParameters()
        atomic = AtomicPhysics(u)
        lam = atomic.compton_wavelength()
        assert lam > 0
        assert lam < 1e-11

    def test_fine_structure_effects_keys(self):
        u = UniverseParameters()
        atomic = AtomicPhysics(u)
        effects = atomic.fine_structure_effects()
        assert 'a0' in effects
        assert 'a0_norm' in effects
        assert 'a0_over_λc' in effects
        assert 'E_bind' in effects
        assert 'E_bind_norm' in effects
        assert 'a0_over_l_planck' in effects

    def test_bohr_radius_scales_with_alpha(self):
        u_small_alpha = UniverseParameters(alpha=0.001)
        u_large_alpha = UniverseParameters(alpha=0.05)
        a0_small = AtomicPhysics(u_small_alpha).bohr_radius()
        a0_large = AtomicPhysics(u_large_alpha).bohr_radius()
        assert a0_small > a0_large


# ==================== NuclearPhysics ====================


class TestNuclearPhysics:
    def test_qcd_scale_in_range(self):
        u = UniverseParameters()
        nuc = NuclearPhysics(u)
        lam = nuc.qcd_scale()
        assert 0.5e-28 < lam < 1e-27

    def test_binding_energy_fe56_positive(self):
        u = UniverseParameters()
        nuc = NuclearPhysics(u)
        B = nuc.binding_energy(56, 26)
        assert B > 0

    def test_binding_per_nucleon_fe56(self):
        u = UniverseParameters()
        nuc = NuclearPhysics(u)
        bpn = nuc.binding_per_nucleon(56, 26)
        assert 7 < bpn < 10

    def test_coulomb_barrier_pp(self):
        u = UniverseParameters()
        nuc = NuclearPhysics(u)
        barrier = nuc.coulomb_barrier(1, 1, 1, 1)
        assert barrier > 0

    def test_neutron_drip_line_returns_int(self):
        u = UniverseParameters()
        nuc = NuclearPhysics(u)
        N = nuc.neutron_drip_line(26)
        assert isinstance(N, int)
        assert N > 26

    def test_binding_energy_scales_with_alpha(self):
        u_low = UniverseParameters(alpha=0.001)
        u_high = UniverseParameters(alpha=0.1)
        nuc_low = NuclearPhysics(u_low)
        nuc_high = NuclearPhysics(u_high)
        bpn_low = nuc_low.binding_per_nucleon(56, 26)
        bpn_high = nuc_high.binding_per_nucleon(56, 26)
        assert bpn_high < bpn_low
