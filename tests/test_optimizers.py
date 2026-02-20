"""Тесты для оптимизаторов пригодности вселенных."""

import importlib.util
import sys
from pathlib import Path

# Использовать non-interactive бэкенд до импорта matplotlib в оптимизаторах
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
import numpy as np

from multiverse_tester import (
    UniverseParameters,
    UniverseAnalyzer,
    UniversalConstants,
    ALPHA_OUR,
)


def _load_optimizer_module(name: str, filename: str):
    """Динамически загружает модуль оптимизатора (имя файла может начинаться с цифры)."""
    root = Path(__file__).resolve().parent.parent
    path = root / filename
    if not path.exists():
        pytest.skip(f"Оптимизатор {filename} не найден")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ==================== 2D ОПТИМИЗАТОР ====================


class Test2DOptimizer:
    """Тесты для UniverseOptimizer (2D оптимизатор)."""

    @pytest.fixture
    def opt2d(self):
        mod = _load_optimizer_module("opt2d", "2Doptimizator.py")
        return mod.UniverseOptimizer()

    def test_objective_function_returns_float(self, opt2d):
        """Целевая функция возвращает число в [0, 1]."""
        obj = opt2d.objective_function(ALPHA_OUR, m_p_ratio=1.0, verbose=False)
        assert isinstance(obj, float)
        assert 0 <= obj <= 1.1  # 1 - score, score ~0..1

    def test_objective_function_our_universe(self, opt2d):
        """Наша вселенная даёт приемлемый score (не максимальная непригодность)."""
        obj = opt2d.objective_function(ALPHA_OUR, m_p_ratio=1.0, verbose=False)
        assert obj < 1.0, "Наша вселенная не должна быть полностью непригодной"

    def test_objective_function_extreme_alpha(self, opt2d):
        """Экстремальные α дают высокую целевую функцию (низкая пригодность)."""
        obj_small = opt2d.objective_function(1 / 500, m_p_ratio=1.0, verbose=False)
        obj_large = opt2d.objective_function(1 / 20, m_p_ratio=1.0, verbose=False)
        assert obj_small > 0
        assert obj_large > 0

    def test_objective_function_populates_history(self, opt2d):
        """Вызов objective_function добавляет запись в optimization_history."""
        opt2d.optimization_history = []
        opt2d.objective_function(0.008, m_p_ratio=1.0, verbose=False)
        assert len(opt2d.optimization_history) == 1
        h = opt2d.optimization_history[0]
        assert h["alpha"] == 0.008
        assert h["m_p_ratio"] == 1.0
        assert "score" in h
        assert "index" in h
        assert "metrics" in h

    def test_optimize_alpha_returns_dict(self, opt2d):
        """optimize_alpha возвращает словарь с ожидаемыми ключами."""
        result = opt2d.optimize_alpha(
            bounds=(1 / 200, 1 / 50),
            method="brent",
            verbose=False,
        )
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "score" in result
        assert "index" in result
        assert "metrics" in result
        assert "history" in result

    def test_optimize_alpha_optimal_in_bounds(self, opt2d):
        """Оптимальная α лежит в заданных границах."""
        bounds = (1 / 200, 1 / 50)
        result = opt2d.optimize_alpha(bounds=bounds, method="brent", verbose=False)
        alpha = result["alpha"]
        assert bounds[0] <= alpha <= bounds[1]

    def test_optimize_alpha_score_in_range(self, opt2d):
        """Score оптимума в диапазоне [0, 1]."""
        result = opt2d.optimize_alpha(bounds=(1 / 200, 1 / 50), method="brent", verbose=False)
        assert 0 <= result["score"] <= 1

    def test_optimize_2d_returns_dict(self, opt2d):
        """optimize_2d возвращает словарь с ожидаемыми ключами."""
        result = opt2d.optimize_2d(
            alpha_bounds=(1 / 200, 1 / 50),
            m_p_bounds=(0.7, 1.3),
            popsize=5,
            maxiter=3,
            verbose=False,
        )
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "m_p_ratio" in result
        assert "score" in result
        assert "index" in result
        assert "metrics" in result
        assert "nucleosynthesis" in result
        assert "history" in result
        assert "success" in result

    def test_optimize_2d_optimal_in_bounds(self, opt2d):
        """Оптимальные α и m_p лежат в границах."""
        result = opt2d.optimize_2d(
            alpha_bounds=(1 / 200, 1 / 50),
            m_p_bounds=(0.7, 1.3),
            popsize=5,
            maxiter=3,
            verbose=False,
        )
        assert 1 / 200 <= result["alpha"] <= 1 / 50
        assert 0.7 <= result["m_p_ratio"] <= 1.3

    def test_grid_search_returns_dict(self, opt2d):
        """grid_search возвращает словарь с ожидаемой структурой."""
        result = opt2d.grid_search(
            alpha_points=5,
            m_p_points=4,
            alpha_range=(1 / 200, 1 / 50),
            m_p_range=(0.7, 1.3),
        )
        assert isinstance(result, dict)
        assert "alphas" in result
        assert "m_p_ratios" in result
        assert "score_map" in result
        assert "category_map" in result
        assert "best_alpha" in result
        assert "best_m_p" in result
        assert "best_score" in result

    def test_grid_search_shapes(self, opt2d):
        """grid_search возвращает массивы правильной формы."""
        result = opt2d.grid_search(
            alpha_points=5,
            m_p_points=4,
            alpha_range=(1 / 200, 1 / 50),
            m_p_range=(0.7, 1.3),
        )
        assert len(result["alphas"]) == 5
        assert len(result["m_p_ratios"]) == 4
        assert result["score_map"].shape == (5, 4)
        assert result["category_map"].shape == (5, 4)

    def test_grid_search_best_in_grid(self, opt2d):
        """Лучшая точка совпадает с максимумом score_map."""
        result = opt2d.grid_search(
            alpha_points=5,
            m_p_points=4,
            alpha_range=(1 / 200, 1 / 50),
            m_p_range=(0.7, 1.3),
        )
        max_idx = np.unravel_index(np.argmax(result["score_map"]), result["score_map"].shape)
        assert result["best_alpha"] == result["alphas"][max_idx[0]]
        assert result["best_m_p"] == result["m_p_ratios"][max_idx[1]]
        assert result["best_score"] == result["score_map"][max_idx]


# ==================== run_all_optimizers (2D) ====================


class TestRunAllOptimizers2D:
    """Тесты для run_2d_optimizer из run_all_optimizers."""

    def test_run_2d_optimizer_returns_dict(self):
        mod = _load_optimizer_module("run_all", "run_all_optimizers.py")
        # Подавляем вывод и сохранение графиков
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            results = mod.run_2d_optimizer()

        assert isinstance(results, dict)
        assert "opt_alpha" in results
        assert "opt_alpha_score" in results
        assert "opt_alpha_2d" in results
        assert "opt_m_p" in results
        assert "opt_2d_score" in results
        assert "our_score" in results

    def test_run_2d_opt_alpha_in_range(self):
        mod = _load_optimizer_module("run_all", "run_all_optimizers.py")
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            results = mod.run_2d_optimizer()

        assert 1 / 300 <= results["opt_alpha"] <= 1 / 30
        assert 0 <= results["opt_alpha_score"] <= 1


# ==================== 6D ОПТИМИЗАТОР ====================


class Test6DOptimizer:
    """Тесты для HyperVolume6D (уменьшенная сетка для скорости)."""

    def test_6d_grid_generation(self):
        mod = _load_optimizer_module("opt6d", "6D_optimizator.py")
        hv = mod.HyperVolume6D()
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = hv.generate_6d_grid(points=2)  # 2^6=64 точки

        assert "best_alpha" in result
        assert "best_score" in result
        assert "score_6d" in result
        assert result["score_6d"].shape == (2, 2, 2, 2, 2, 2)


# ==================== ВИЗУАЛИЗАТОР ====================


class TestOptimizationVisualizer:
    """Тесты для OptimizationVisualizer (без отображения графиков)."""

    @pytest.fixture
    def visualizer(self):
        mod = _load_optimizer_module("opt2d", "2Doptimizator.py")
        optimizer = mod.UniverseOptimizer()
        return mod.OptimizationVisualizer(optimizer)

    def test_visualizer_creation(self, visualizer):
        """Визуализатор создаётся с оптимизатором."""
        assert visualizer.opt is not None

    def test_plot_optimization_1d_no_crash(self, visualizer):
        """plot_optimization_1d не падает при валидных данных (matplotlib Agg)."""
        result = visualizer.opt.optimize_alpha(
            bounds=(1 / 200, 1 / 50), method="brent", verbose=False
        )
        visualizer.plot_optimization_1d(result)  # не падает с Agg
        import matplotlib.pyplot as plt
        plt.close("all")
