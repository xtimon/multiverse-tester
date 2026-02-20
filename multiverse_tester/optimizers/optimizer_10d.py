#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10D Гиперобъем пригодности вселенных
Параметры: α, m_p, m_e, G, c, ħ, ε₀, k_B, H₀, Λ (космологическая постоянная)

Λ определяет плотность тёмной энергии и ускорение расширения.
Слишком большая Λ — структуры не успевают сформироваться;
отрицательная Λ — быстрый реколлапс.
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

from multiverse_tester import UniversalConstants, UniverseParameters, UniverseAnalyzer


class HyperVolume10D:
    """10D гиперобъем с Λ. Адаптивная рефайнмент-стратегия."""

    def __init__(self):
        self.const = UniversalConstants()
        self.results = {}

    def _eval_point(self, alpha: float, m_p_r: float, m_e_r: float, G_r: float,
                    c_r: float, hbar_r: float, eps_r: float, k_B_r: float,
                    H0_r: float, Lambda_r: float) -> float:
        try:
            u = UniverseParameters(
                alpha=alpha,
                m_p=m_p_r * self.const.m_p,
                m_e=m_e_r * self.const.m_e,
                G=G_r * self.const.G,
                c=c_r * self.const.c,
                hbar=hbar_r * self.const.hbar,
                epsilon_0=eps_r * self.const.epsilon_0,
                k_B=k_B_r * self.const.k_B,
                H_0=H0_r * self.const.H_0,
                Lambda=Lambda_r * self.const.Lambda
            )
            analyzer = UniverseAnalyzer(u)
            _, score, _ = analyzer.calculate_habitability_index()
            return score
        except Exception:
            return 0.0

    def generate_10d_adaptive(self,
                             alpha_range: Tuple[float, float] = (1/400, 1/15),
                             m_p_range: Tuple[float, float] = (0.1, 5.0),
                             m_e_range: Tuple[float, float] = (0.1, 5.0),
                             G_range: Tuple[float, float] = (0.05, 10.0),
                             c_range: Tuple[float, float] = (0.2, 3.0),
                             hbar_range: Tuple[float, float] = (0.2, 3.0),
                             epsilon_0_range: Tuple[float, float] = (0.1, 5.0),
                             k_B_range: Tuple[float, float] = (0.1, 5.0),
                             H0_range: Tuple[float, float] = (0.2, 5.0),
                             Lambda_range: Tuple[float, float] = (0.1, 10.0),
                             coarse_points: int = 3,
                             zoom_points: int = 2,
                             zoom_fraction: float = 0.25,
                             max_refinements: int = 2) -> Dict:
        """Адаптивный поиск в 10D."""
        def _run_grid(ranges: List[Tuple[float, float]], pts: int) -> Tuple[List, np.ndarray, float, List[float]]:
            grids = [np.linspace(r[0], r[1], pts) for r in ranges]
            total = pts ** 10
            scores = np.zeros(total)
            idx = 0
            for i0 in range(pts):
                for i1 in range(pts):
                    for i2 in range(pts):
                        for i3 in range(pts):
                            for i4 in range(pts):
                                for i5 in range(pts):
                                    for i6 in range(pts):
                                        for i7 in range(pts):
                                            for i8 in range(pts):
                                                for i9 in range(pts):
                                                    scores[idx] = self._eval_point(
                                                        grids[0][i0], grids[1][i1], grids[2][i2],
                                                        grids[3][i3], grids[4][i4], grids[5][i5],
                                                        grids[6][i6], grids[7][i7], grids[8][i8],
                                                        grids[9][i9]
                                                    )
                                                    idx += 1
                                                    if idx % 5000 == 0:
                                                        print(f"   Прогресс: {idx}/{total} ({100*idx/total:.1f}%)")
            best_idx = np.argmax(scores)
            multi_idx = np.unravel_index(best_idx, (pts,) * 10)
            best_coords = [grids[d][multi_idx[d]] for d in range(10)]
            return grids, scores, float(scores[best_idx]), best_coords

        def _zoom_ranges(best: List[float], ranges: List[Tuple[float, float]], frac: float) -> List[Tuple[float, float]]:
            return [
                (max(r[0], b - frac * (r[1] - r[0])), min(r[1], b + frac * (r[1] - r[0])))
                for (r, b) in zip(ranges, best)
            ]

        full_ranges = [alpha_range, m_p_range, m_e_range, G_range, c_range, hbar_range,
                      epsilon_0_range, k_B_range, H0_range, Lambda_range]
        labels = ['α', 'm_p', 'm_e', 'G', 'c', 'ħ', 'ε₀', 'k_B', 'H₀', 'Λ']

        print(f"\nАДАПТИВНЫЙ 10D: Фаза 1 — грубая сетка {coarse_points}^10 = {coarse_points**10} точек")
        for lbl, (lo, hi) in zip(labels, full_ranges):
            print(f"   {lbl}: [{lo:.4f}, {hi:.4f}]")

        grids1, scores1, best_score, best_coords = _run_grid(full_ranges, coarse_points)

        self.results['score_10d'] = scores1.reshape((coarse_points,) * 10)
        for d, key in enumerate(['alphas', 'm_p_ratios', 'm_e_ratios', 'G_ratios', 'c_ratios',
                                 'hbar_ratios', 'eps_ratios', 'k_B_ratios', 'H0_ratios', 'Lambda_ratios']):
            self.results[key] = grids1[d]

        for ref in range(max_refinements - 1):
            zoom_ranges = _zoom_ranges(best_coords, full_ranges, zoom_fraction)
            total_zoom = zoom_points ** 10
            print(f"\n   Фаза {ref+2}: зум ({zoom_points}^10 = {total_zoom} точек)")
            _, _, zoom_score, zoom_coords = _run_grid(zoom_ranges, zoom_points)
            if zoom_score > best_score:
                best_score = zoom_score
                best_coords = zoom_coords
                print(f"   → Новый оптимум: score = {best_score:.4f}")
            else:
                print(f"   → Оптимум стабилен (zoom score = {zoom_score:.4f})")

        print(f"\nОПТИМУМ (10D):")
        print(f"   α={best_coords[0]:.4f}, m_p={best_coords[1]:.2f}, m_e={best_coords[2]:.2f}, G={best_coords[3]:.2f}")
        print(f"   c={best_coords[4]:.2f}, ħ={best_coords[5]:.2f}, ε₀={best_coords[6]:.2f}, k_B={best_coords[7]:.2f}")
        print(f"   H₀={best_coords[8]:.2f}, Λ/Λ₀={best_coords[9]:.2f}  |  score={best_score:.3f}")

        self.results.update({
            'best_alpha': best_coords[0], 'best_m_p': best_coords[1], 'best_m_e': best_coords[2],
            'best_G': best_coords[3], 'best_c': best_coords[4], 'best_hbar': best_coords[5],
            'best_eps': best_coords[6], 'best_k_B': best_coords[7], 'best_H0': best_coords[8],
            'best_Lambda': best_coords[9], 'best_score': best_score,
        })
        return self.results

    def calculate_10d_volume(self, threshold: float = 0.6) -> Dict:
        if not self.results or 'score_10d' not in self.results:
            return {}
        score = self.results['score_10d']
        frac = (score > threshold).sum() / score.size
        print(f"\n10D ГИПЕРОБЪЕМ (score > {threshold}): доля {frac*100:.4f}%")
        return {'fraction': float(frac)}


def main():
    print("="*70)
    print("10D: α, m_p, m_e, G, c, ħ, ε₀, k_B, H₀, Λ")
    print("="*70)
    hv = HyperVolume10D()
    results = hv.generate_10d_adaptive(
        alpha_range=(1/400, 1/15),
        m_p_range=(0.1, 5.0),
        m_e_range=(0.1, 5.0),
        G_range=(0.05, 10.0),
        c_range=(0.2, 3.0),
        hbar_range=(0.2, 3.0),
        epsilon_0_range=(0.1, 5.0),
        k_B_range=(0.1, 5.0),
        H0_range=(0.2, 5.0),
        Lambda_range=(0.1, 10.0),
        coarse_points=3,
        zoom_points=2,
        zoom_fraction=0.25,
        max_refinements=2
    )
    hv.calculate_10d_volume(threshold=0.6)
    our = UniverseParameters()
    _, our_score, _ = UniverseAnalyzer(our).calculate_habitability_index()
    print(f"\nНаша Вселенная: score = {our_score:.3f}")
    print(f"Оптимум (10D): score = {results['best_score']:.3f}")


if __name__ == "__main__":
    main()
