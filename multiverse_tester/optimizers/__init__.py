# -*- coding: utf-8 -*-
"""
Оптимизаторы пригодности вселенных (2D–10D).
"""

from multiverse_tester.optimizers.optimizer_2d import (
    UniverseOptimizer,
    OptimizationVisualizer,
    score_to_category,
)
from multiverse_tester.optimizers.optimizer_3d import Landscape3D
from multiverse_tester.optimizers.optimizer_4d import HyperVolume4D, Visualizer4D
from multiverse_tester.optimizers.optimizer_5d import HyperVolume5D, Visualizer5D
from multiverse_tester.optimizers.optimizer_6d import HyperVolume6D, Visualizer6D
from multiverse_tester.optimizers.optimizer_7d import HyperVolume7D
from multiverse_tester.optimizers.optimizer_8d import HyperVolume8D
from multiverse_tester.optimizers.optimizer_9d import HyperVolume9D
from multiverse_tester.optimizers.optimizer_10d import HyperVolume10D

from multiverse_tester.optimizers.optimizer_base import (
    generate_nd_adaptive,
    plot_nd_2d_slice,
)

__all__ = [
    'UniverseOptimizer',
    'OptimizationVisualizer',
    'score_to_category',
    'Landscape3D',
    'HyperVolume4D',
    'Visualizer4D',
    'HyperVolume5D',
    'Visualizer5D',
    'HyperVolume6D',
    'Visualizer6D',
    'HyperVolume7D',
    'HyperVolume8D',
    'HyperVolume9D',
    'HyperVolume10D',
    'generate_nd_adaptive',
    'plot_nd_2d_slice',
]
