#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
8D Гиперобъем пригодности вселенных
Параметры: α, m_p, m_e, G, c, ħ, ε₀, k_B (постоянная Больцмана)

k_B связывает температурную шкалу с энергией: E = k_B·T
Используется в термоядерных реакциях и звёздной физике.
"""

import numpy as np
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

from multiverse_tester import UniversalConstants, UniverseParameters, UniverseAnalyzer


class HyperVolume8D:
    """
    8D гиперобъем пригодности вселенных с k_B (постоянная Больцмана).
    Адаптивная рефайнмент-стратегия для 8-мерного пространства.
    """
    
    def __init__(self):
        self.const = UniversalConstants()
        self.results = {}
    
    def _eval_point(self, alpha: float, m_p_r: float, m_e_r: float, G_r: float,
                    c_r: float, hbar_r: float, eps_r: float, k_B_r: float) -> float:
        """Оценка одной точки параметров"""
        try:
            u = UniverseParameters(
                alpha=alpha,
                m_p=m_p_r * self.const.m_p,
                m_e=m_e_r * self.const.m_e,
                G=G_r * self.const.G,
                c=c_r * self.const.c,
                hbar=hbar_r * self.const.hbar,
                epsilon_0=eps_r * self.const.epsilon_0,
                k_B=k_B_r * self.const.k_B
            )
            analyzer = UniverseAnalyzer(u)
            _, score, _ = analyzer.calculate_habitability_index()
            return score
        except Exception:
            return 0.0
    
    def generate_8d_adaptive(self,
                            alpha_range: Tuple[float, float] = (1/400, 1/15),
                            m_p_range: Tuple[float, float] = (0.1, 5.0),
                            m_e_range: Tuple[float, float] = (0.1, 5.0),
                            G_range: Tuple[float, float] = (0.05, 10.0),
                            c_range: Tuple[float, float] = (0.2, 3.0),
                            hbar_range: Tuple[float, float] = (0.2, 3.0),
                            epsilon_0_range: Tuple[float, float] = (0.1, 5.0),
                            k_B_range: Tuple[float, float] = (0.1, 5.0),
                            coarse_points: int = 3,
                            zoom_points: int = 3,
                            zoom_fraction: float = 0.25,
                            max_refinements: int = 2) -> Dict:
        """
        Адаптивный поиск в 8D: грубая сетка → зум вокруг оптимума.
        coarse_points^8 и zoom_points^8 точек.
        """
        def _run_grid(ranges: List[Tuple[float, float]], pts: int) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
            (ar, mpr, mer, Gr, cr, hbr, epsr, kBr) = ranges
            grids = [
                np.linspace(ar[0], ar[1], pts),
                np.linspace(mpr[0], mpr[1], pts),
                np.linspace(mer[0], mer[1], pts),
                np.linspace(Gr[0], Gr[1], pts),
                np.linspace(cr[0], cr[1], pts),
                np.linspace(hbr[0], hbr[1], pts),
                np.linspace(epsr[0], epsr[1], pts),
                np.linspace(kBr[0], kBr[1], pts),
            ]
            total = pts ** 8
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
                                            scores[idx] = self._eval_point(
                                                grids[0][i0], grids[1][i1], grids[2][i2],
                                                grids[3][i3], grids[4][i4], grids[5][i5],
                                                grids[6][i6], grids[7][i7]
                                            )
                                            idx += 1
                                            if idx % 2000 == 0:
                                                print(f"   Прогресс: {idx}/{total} ({100*idx/total:.1f}%)")
            best_idx = np.argmax(scores)
            multi_idx = np.unravel_index(best_idx, (pts,) * 8)
            best_coords = [grids[d][multi_idx[d]] for d in range(8)]
            return grids, scores, float(scores[best_idx]), best_coords
        
        def _zoom_ranges(best: List[float], ranges: List[Tuple[float, float]], frac: float) -> List[Tuple[float, float]]:
            return [
                (max(r[0], b - frac * (r[1] - r[0])), min(r[1], b + frac * (r[1] - r[0])))
                for (r, b) in zip(ranges, best)
            ]
        
        full_ranges = [alpha_range, m_p_range, m_e_range, G_range, c_range, hbar_range, epsilon_0_range, k_B_range]
        labels = ['α', 'm_p', 'm_e', 'G', 'c', 'ħ', 'ε₀', 'k_B']
        
        print(f"\n🔮 АДАПТИВНЫЙ 8D: Фаза 1 — грубая сетка {coarse_points}^8 = {coarse_points**8} точек")
        for lbl, (lo, hi) in zip(labels, full_ranges):
            print(f"   {lbl}: [{lo:.4f}, {hi:.4f}]")
        
        grids1, scores1, best_score, best_coords = _run_grid(full_ranges, coarse_points)
        
        self.results['score_8d'] = scores1.reshape((coarse_points,) * 8)
        for d, key in enumerate(['alphas', 'm_p_ratios', 'm_e_ratios', 'G_ratios', 'c_ratios', 'hbar_ratios', 'eps_ratios', 'k_B_ratios']):
            self.results[key] = grids1[d]
        
        for ref in range(max_refinements - 1):
            zoom_ranges = _zoom_ranges(best_coords, full_ranges, zoom_fraction)
            total_zoom = zoom_points ** 8
            print(f"\n   Фаза {ref+2}: зум вокруг оптимума ({zoom_points}^8 = {total_zoom} точек)")
            
            _, scores_zoom, zoom_score, zoom_coords = _run_grid(zoom_ranges, zoom_points)
            
            if zoom_score > best_score:
                best_score = zoom_score
                best_coords = zoom_coords
                print(f"   → Новый оптимум: score = {best_score:.4f}")
            else:
                print(f"   → Оптимум стабилен (zoom score = {zoom_score:.4f})")
        
        print(f"\n✅ АДАПТИВНЫЙ ОПТИМУМ (8D):")
        print(f"   α = {best_coords[0]:.6f}")
        print(f"   m_p/m_p₀ = {best_coords[1]:.3f}")
        print(f"   m_e/m_e₀ = {best_coords[2]:.3f}")
        print(f"   G/G₀ = {best_coords[3]:.3f}")
        print(f"   c/c₀ = {best_coords[4]:.3f}")
        print(f"   ħ/ħ₀ = {best_coords[5]:.3f}")
        print(f"   ε₀/ε₀₀ = {best_coords[6]:.3f}")
        print(f"   k_B/k_B₀ = {best_coords[7]:.3f}")
        print(f"   Индекс пригодности = {best_score:.3f}")
        
        self.results.update({
            'best_alpha': best_coords[0], 'best_m_p': best_coords[1], 'best_m_e': best_coords[2],
            'best_G': best_coords[3], 'best_c': best_coords[4], 'best_hbar': best_coords[5],
            'best_eps': best_coords[6], 'best_k_B': best_coords[7], 'best_score': best_score,
        })
        
        return self.results
    
    def calculate_8d_volume(self, threshold: float = 0.6) -> Dict:
        """Вычисляет долю пригодного 8D пространства по грубой сетке"""
        if not self.results or 'score_8d' not in self.results:
            return {}
        score = self.results['score_8d']
        habitable_mask = score > threshold
        voxel_count = int(np.sum(habitable_mask))
        total_voxels = score.size
        volume_fraction = voxel_count / total_voxels
        print(f"\n📊 8D ГИПЕРОБЪЕМ (score > {threshold}):")
        print(f"   Доля пространства: {volume_fraction*100:.4f}%")
        print(f"   Количество точек: {voxel_count}/{total_voxels}")
        return {'fraction': volume_fraction, 'voxel_count': voxel_count}


def main():
    print("="*90)
    print("🌌 8D ГИПЕРОБЪЕМ ПРИГОДНОСТИ ВСЕЛЕННЫХ (α, m_p, m_e, G, c, ħ, ε₀, k_B)")
    print("="*90)
    
    hv = HyperVolume8D()
    results = hv.generate_8d_adaptive(
        alpha_range=(1/400, 1/15),
        m_p_range=(0.1, 5.0),
        m_e_range=(0.1, 5.0),
        G_range=(0.05, 10.0),
        c_range=(0.2, 3.0),
        hbar_range=(0.2, 3.0),
        epsilon_0_range=(0.1, 5.0),
        k_B_range=(0.1, 5.0),
        coarse_points=3,
        zoom_points=3,
        zoom_fraction=0.25,
        max_refinements=2
    )
    
    vol = hv.calculate_8d_volume(threshold=0.6)
    
    our = UniverseParameters(name="🌍 Наша Вселенная")
    _, our_score, _ = UniverseAnalyzer(our).calculate_habitability_index()
    print(f"\n🌍 НАША ВСЕЛЕННАЯ: score = {our_score:.3f}")
    print(f"🌟 ОПТИМУМ (8D): score = {results['best_score']:.3f}")
    if vol:
        print(f"📊 Доля пригодного 8D пространства: {vol['fraction']*100:.2f}%")
    print("\n" + "="*90)


if __name__ == "__main__":
    main()
