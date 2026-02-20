# -*- coding: utf-8 -*-
"""
Базовые утилиты для ND-оптимизаторов: адаптивный поиск и визуализация.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable

from multiverse_tester import UniversalConstants, UniverseParameters, UniverseAnalyzer


def _build_universe_from_ratios(const: UniversalConstants, ratios: List[float], dim: int) -> UniverseParameters:
    """Собирает UniverseParameters из списка отношений (α, m_p_r, m_e_r, G_r, c_r, hbar_r, eps_r[, k_B_r, ...])."""
    kwargs = {
        'alpha': ratios[0],
        'm_p': ratios[1] * const.m_p,
        'm_e': ratios[2] * const.m_e,
        'G': ratios[3] * const.G,
        'c': ratios[4] * const.c,
        'hbar': ratios[5] * const.hbar,
        'epsilon_0': ratios[6] * const.epsilon_0,
    }
    if dim >= 8:
        kwargs['k_B'] = ratios[7] * const.k_B
    if dim >= 9:
        kwargs['H_0'] = ratios[8] * const.H_0
    if dim >= 10:
        kwargs['Lambda'] = ratios[9] * const.Lambda
    return UniverseParameters(**kwargs)


def _default_eval(const: UniversalConstants, dim: int, ratios: List[float]) -> float:
    """Оценка одной точки по списку отношений (α, m_p_r, m_e_r, ...)."""
    try:
        u = _build_universe_from_ratios(const, ratios[:dim], dim)
        _, score, _ = UniverseAnalyzer(u).calculate_habitability_index()
        return score
    except Exception:
        return 0.0


def generate_nd_adaptive(
    const: UniversalConstants,
    dim: int,
    ranges: List[Tuple[float, float]],
    coarse_points: int = 3,
    zoom_points: int = 4,
    zoom_fraction: float = 0.25,
    max_refinements: int = 2,
    score_key: str = 'score_nd',
    eval_fn: Optional[Callable[..., float]] = None,
) -> Dict:
    """
    Адаптивный ND-поиск: грубая сетка → зум вокруг оптимума.
    
    Args:
        const: константы нашей Вселенной
        dim: размерность (7 или 8)
        ranges: список (min, max) для каждого параметра
        coarse_points: точек на грубой фазе
        zoom_points: точек в фазе зума
        zoom_fraction: доля диапазона для зума вокруг оптимума
        max_refinements: число рефайнментов
        score_key: ключ для сохранения массива score в results
        eval_fn: опционально, функция (const, *ratios) -> score; по умолчанию встроенная
    """
    if eval_fn is None:
        def _eval(*ratios):
            return _default_eval(const, dim, list(ratios))
    else:
        _eval = lambda *r: eval_fn(const, *r)

    def _run_grid(ranges_in: List[Tuple[float, float]], pts: int) -> Tuple[List, np.ndarray, float, List[float]]:
        grids = [np.linspace(r[0], r[1], pts) for r in ranges_in]
        total = pts ** dim
        scores = np.zeros(total)
        for idx, multi_idx in enumerate(np.ndindex((pts,) * dim)):
            ratios = [grids[d][multi_idx[d]] for d in range(dim)]
            scores[idx] = _eval(*ratios)
            if idx % 2000 == 0 and total > 2000:
                print(f"   Прогресс: {idx}/{total} ({100*idx/total:.1f}%)")
        best_idx = np.argmax(scores)
        multi_idx = np.unravel_index(best_idx, (pts,) * dim)
        best_coords = [grids[d][multi_idx[d]] for d in range(dim)]
        return grids, scores, float(scores[best_idx]), best_coords

    def _zoom_ranges(best: List[float], ranges_in: List[Tuple[float, float]], frac: float) -> List[Tuple[float, float]]:
        return [
            (max(r[0], b - frac * (r[1] - r[0])), min(r[1], b + frac * (r[1] - r[0])))
            for (r, b) in zip(ranges_in, best)
        ]

    labels = ['α', 'm_p', 'm_e', 'G', 'c', 'ħ', 'ε₀', 'k_B', 'H₀', 'Λ'][:dim]
    result_keys = ['alphas', 'm_p_ratios', 'm_e_ratios', 'G_ratios', 'c_ratios',
                  'hbar_ratios', 'eps_ratios', 'k_B_ratios', 'H0_ratios', 'Lambda_ratios'][:dim]
    best_keys = ['best_alpha', 'best_m_p', 'best_m_e', 'best_G', 'best_c',
                 'best_hbar', 'best_eps', 'best_k_B', 'best_H0', 'best_Lambda'][:dim]

    print(f"\nАДАПТИВНЫЙ {dim}D: Фаза 1 — грубая сетка {coarse_points}^{dim} = {coarse_points**dim} точек")
    for lbl, (lo, hi) in zip(labels, ranges):
        print(f"   {lbl}: [{lo:.4f}, {hi:.4f}]")

    grids1, scores1, best_score, best_coords = _run_grid(ranges, coarse_points)
    results = {
        score_key: scores1.reshape((coarse_points,) * dim),
        **{result_keys[d]: grids1[d] for d in range(dim)},
    }

    for ref in range(max_refinements - 1):
        zoom_ranges = _zoom_ranges(best_coords, ranges, zoom_fraction)
        total_zoom = zoom_points ** dim
        print(f"\n   Фаза {ref+2}: зум ({zoom_points}^{dim} = {total_zoom} точек)")
        _, _, zoom_score, zoom_coords = _run_grid(zoom_ranges, zoom_points)
        if zoom_score > best_score:
            best_score = zoom_score
            best_coords = zoom_coords
            print(f"   → Новый оптимум: score = {best_score:.4f}")
        else:
            print(f"   → Оптимум стабилен (zoom score = {zoom_score:.4f})")

    print(f"\nОПТИМУМ ({dim}D): score = {best_score:.3f}")
    results.update({
        **{best_keys[d]: best_coords[d] for d in range(dim)},
        'best_score': best_score,
    })
    return results


def plot_nd_2d_slice(
    results: Dict,
    score_key: str,
    dim: int,
    fig_dir: Optional[str] = None,
) -> None:
    """
    Строит 2D срез (α, m_p) из ND results и сохраняет в файл.
    
    Args:
        results: словарь с score_key (ND массив), alphas, m_p_ratios
        score_key: ключ массива пригодности
        dim: размерность
        fig_dir: папка для сохранения (если None — plt.show())
    """
    if score_key not in results:
        return
    score_nd = results[score_key]
    alphas = results.get('alphas')
    m_p_ratios = results.get('m_p_ratios')
    if alphas is None or m_p_ratios is None:
        return
    # Берём средний срез по остальным осям (все индексы = n//2)
    slice_2d = score_nd
    for _ in range(dim - 2):
        mid = slice_2d.shape[-1] // 2
        slice_2d = np.take(slice_2d, mid, axis=-1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        slice_2d.T,
        aspect='auto',
        extent=[alphas[0], alphas[-1], m_p_ratios[0], m_p_ratios[-1]],
        origin='lower',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel('α')
    ax.set_ylabel('m_p / m_p₀')
    ax.set_title(f'{dim}D Slice (α, m_p)')
    plt.colorbar(im, ax=ax, label='Habitability')
    plt.tight_layout()
    if fig_dir:
        from pathlib import Path
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / f'fig_{score_key}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
