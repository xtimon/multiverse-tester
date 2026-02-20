"""
MultiverseTester - Симуляция пригодности вселенных для жизни.

Исследует, как различные значения фундаментальных физических констант
влияют на возможность возникновения жизни в мультивселенной.

Author: Timur Isanov
Email: tisanov@yahoo.com
"""

__version__ = "1.0.0"

from multiverse_tester.core import (
    ALPHA_OUR,
    UniversalConstants,
    UniverseParameters,
    AtomicPhysics,
    NuclearPhysics,
    StellarNucleosynthesis,
    HabitabilityIndex,
    UniverseAnalyzer,
    MultiverseDynamicsExplorer,
)

from multiverse_tester.optimizers import (
    UniverseOptimizer,
    HyperVolume4D,
    HyperVolume5D,
    HyperVolume6D,
    HyperVolume7D,
    HyperVolume8D,
    HyperVolume9D,
    HyperVolume10D,
)

__all__ = [
    "__version__",
    "ALPHA_OUR",
    "UniversalConstants",
    "UniverseParameters",
    "AtomicPhysics",
    "NuclearPhysics",
    "StellarNucleosynthesis",
    "HabitabilityIndex",
    "UniverseAnalyzer",
    "MultiverseDynamicsExplorer",
    "UniverseOptimizer",
    "HyperVolume4D",
    "HyperVolume5D",
    "HyperVolume6D",
    "HyperVolume7D",
    "HyperVolume8D",
    "HyperVolume9D",
    "HyperVolume10D",
]
