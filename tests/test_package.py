"""Тесты импорта пакета и CLI."""

import pytest
from multiverse_tester import (
    __version__,
    UniversalConstants,
    UniverseParameters,
    AtomicPhysics,
    NuclearPhysics,
    StellarNucleosynthesis,
    HabitabilityIndex,
    UniverseAnalyzer,
    MultiverseDynamicsExplorer,
)


class TestPackageImport:
    def test_version_format(self):
        parts = __version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)

    def test_all_exports_importable(self):
        assert UniversalConstants is not None
        assert UniverseParameters is not None
        assert AtomicPhysics is not None
        assert NuclearPhysics is not None
        assert StellarNucleosynthesis is not None
        assert HabitabilityIndex is not None
        assert UniverseAnalyzer is not None
        assert MultiverseDynamicsExplorer is not None


class TestCLI:
    def test_cli_main_importable(self):
        from multiverse_tester.cli import main
        assert callable(main)
