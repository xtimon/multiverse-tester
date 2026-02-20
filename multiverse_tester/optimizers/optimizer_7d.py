#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
7D Гиперобъем пригодности вселенных
Параметры: α, m_p, m_e, G, c, ħ, ε₀ (диэлектрическая проницаемость вакуума)

Использует адаптивную рефайнмент-стратегию:
- Фаза 1: грубая сетка по всему пространству
- Фаза 2+: зум вокруг найденного оптимума с более плотной сеткой
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from multiverse_tester import UniversalConstants, UniverseParameters, UniverseAnalyzer
from multiverse_tester.optimizers.optimizer_base import generate_nd_adaptive


class HyperVolume7D:
    """
    7D гиперобъем пригодности вселенных
    ε₀ определяет силу электромагнитного взаимодействия через α = e²/(4π ε₀ ℏ c)
    Поддерживает адаптивную рефайнмент-стратегию для точного поиска оптимума.
    """
    
    def __init__(self):
        self.const = UniversalConstants()
        self.results = {}
    
    def _eval_point(self, alpha: float, m_p_r: float, m_e_r: float, G_r: float,
                    c_r: float, hbar_r: float, eps_r: float) -> float:
        """Оценка одной точки параметров"""
        try:
            u = UniverseParameters(
                alpha=alpha,
                m_p=m_p_r * self.const.m_p,
                m_e=m_e_r * self.const.m_e,
                G=G_r * self.const.G,
                c=c_r * self.const.c,
                hbar=hbar_r * self.const.hbar,
                epsilon_0=eps_r * self.const.epsilon_0
            )
            analyzer = UniverseAnalyzer(u)
            _, score, _ = analyzer.calculate_habitability_index()
            return score
        except Exception:
            return 0.0
        
    def generate_7d_grid(self, 
                         alpha_range: Tuple[float, float] = (1/400, 1/15),
                         m_p_range: Tuple[float, float] = (0.1, 5.0),
                         m_e_range: Tuple[float, float] = (0.1, 5.0),
                         G_range: Tuple[float, float] = (0.05, 10.0),
                         c_range: Tuple[float, float] = (0.2, 3.0),
                         hbar_range: Tuple[float, float] = (0.2, 3.0),
                         epsilon_0_range: Tuple[float, float] = (0.1, 5.0),
                         points: int = 5) -> Dict:
        """
        Генерирует 7D сетку пригодности
        points^7 = 5^7 = 78,125 точек (расширенная сетка)
        """
        print(f"\n🔮 ГЕНЕРАЦИЯ 7D ГИПЕРОБЪЕМА {points}^7")
        print(f"   α: [{alpha_range[0]:.4f}, {alpha_range[1]:.4f}]")
        print(f"   m_p/m_p₀: [{m_p_range[0]:.2f}, {m_p_range[1]:.2f}]")
        print(f"   m_e/m_e₀: [{m_e_range[0]:.2f}, {m_e_range[1]:.2f}]")
        print(f"   G/G₀: [{G_range[0]:.2f}, {G_range[1]:.2f}]")
        print(f"   c/c₀: [{c_range[0]:.2f}, {c_range[1]:.2f}]")
        print(f"   ħ/ħ₀: [{hbar_range[0]:.2f}, {hbar_range[1]:.2f}]")
        print(f"   ε₀/ε₀₀: [{epsilon_0_range[0]:.2f}, {epsilon_0_range[1]:.2f}]")
        
        alphas = np.linspace(alpha_range[0], alpha_range[1], points)
        m_p_ratios = np.linspace(m_p_range[0], m_p_range[1], points)
        m_e_ratios = np.linspace(m_e_range[0], m_e_range[1], points)
        G_ratios = np.linspace(G_range[0], G_range[1], points)
        c_ratios = np.linspace(c_range[0], c_range[1], points)
        hbar_ratios = np.linspace(hbar_range[0], hbar_range[1], points)
        eps_ratios = np.linspace(epsilon_0_range[0], epsilon_0_range[1], points)
        
        score_7d = np.zeros((points,) * 7)
        total_points = points ** 7
        count = 0
        
        for i, alpha in enumerate(alphas):
            for j, m_p_ratio in enumerate(m_p_ratios):
                for k, m_e_ratio in enumerate(m_e_ratios):
                    for l, G_ratio in enumerate(G_ratios):
                        for m, c_ratio in enumerate(c_ratios):
                            for n, hbar_ratio in enumerate(hbar_ratios):
                                for o, eps_ratio in enumerate(eps_ratios):
                                    score_7d[i, j, k, l, m, n, o] = self._eval_point(
                                        alpha, m_p_ratio, m_e_ratio, G_ratio,
                                        c_ratio, hbar_ratio, eps_ratio
                                    )
                                    
                                    count += 1
                                    if count % 5000 == 0:
                                        pct = count / total_points * 100
                                        print(f"   Прогресс: {count}/{total_points} ({pct:.1f}%)")
        
        max_idx = np.unravel_index(np.argmax(score_7d), score_7d.shape)
        best_alpha = alphas[max_idx[0]]
        best_m_p = m_p_ratios[max_idx[1]]
        best_m_e = m_e_ratios[max_idx[2]]
        best_G = G_ratios[max_idx[3]]
        best_c = c_ratios[max_idx[4]]
        best_hbar = hbar_ratios[max_idx[5]]
        best_eps = eps_ratios[max_idx[6]]
        best_score = score_7d[max_idx]
        
        print(f"\n✅ ГЛОБАЛЬНЫЙ ОПТИМУМ (7D):")
        print(f"   α = {best_alpha:.6f}")
        print(f"   m_p/m_p₀ = {best_m_p:.3f}")
        print(f"   m_e/m_e₀ = {best_m_e:.3f}")
        print(f"   G/G₀ = {best_G:.3f}")
        print(f"   c/c₀ = {best_c:.3f}")
        print(f"   ħ/ħ₀ = {best_hbar:.3f}")
        print(f"   ε₀/ε₀₀ = {best_eps:.3f}")
        print(f"   Индекс пригодности = {best_score:.3f}")
        
        self.results = {
            'alphas': alphas,
            'm_p_ratios': m_p_ratios,
            'm_e_ratios': m_e_ratios,
            'G_ratios': G_ratios,
            'c_ratios': c_ratios,
            'hbar_ratios': hbar_ratios,
            'eps_ratios': eps_ratios,
            'score_7d': score_7d,
            'best_alpha': best_alpha,
            'best_m_p': best_m_p,
            'best_m_e': best_m_e,
            'best_G': best_G,
            'best_c': best_c,
            'best_hbar': best_hbar,
            'best_eps': best_eps,
            'best_score': best_score
        }
        
        return self.results
    
    def generate_7d_adaptive(
        self,
        alpha_range: Tuple[float, float] = (1 / 400, 1 / 15),
        m_p_range: Tuple[float, float] = (0.1, 5.0),
        m_e_range: Tuple[float, float] = (0.1, 5.0),
        G_range: Tuple[float, float] = (0.05, 10.0),
        c_range: Tuple[float, float] = (0.2, 3.0),
        hbar_range: Tuple[float, float] = (0.2, 3.0),
        epsilon_0_range: Tuple[float, float] = (0.1, 5.0),
        coarse_points: int = 3,
        zoom_points: int = 5,
        zoom_fraction: float = 0.25,
        max_refinements: int = 2,
    ) -> Dict:
        """
        Адаптивный поиск: грубая сетка → рефайнмент вокруг лучших точек.
        Делегирует в optimizer_base.generate_nd_adaptive.
        """
        ranges = [
            alpha_range,
            m_p_range,
            m_e_range,
            G_range,
            c_range,
            hbar_range,
            epsilon_0_range,
        ]
        self.results = generate_nd_adaptive(
            self.const,
            dim=7,
            ranges=ranges,
            coarse_points=coarse_points,
            zoom_points=zoom_points,
            zoom_fraction=zoom_fraction,
            max_refinements=max_refinements,
            score_key='score_7d',
        )
        return self.results
    
    def calculate_7d_volume(self, threshold: float = 0.6) -> Dict:
        """Вычисляет 7D гиперобъем пригодного пространства"""
        if not self.results:
            print("❌ Сначала сгенерируйте 7D сетку!")
            return {}
        
        score = self.results['score_7d']
        habitable_mask = score > threshold
        voxel_count = np.sum(habitable_mask)
        total_voxels = score.size
        volume_fraction = voxel_count / total_voxels
        
        print(f"\n📊 7D ГИПЕРОБЪЕМ (score > {threshold}):")
        print(f"   Доля пространства: {volume_fraction*100:.4f}%")
        print(f"   Количество точек: {voxel_count}/{total_voxels}")
        
        return {
            'fraction': volume_fraction,
            'voxel_count': voxel_count,
            'mask': habitable_mask
        }


def main():
    """Запуск 7D анализа"""
    
    print("="*90)
    print("🌌 7D ГИПЕРОБЪЕМ ПРИГОДНОСТИ ВСЕЛЕННЫХ v1.0")
    print("="*90)
    print("\n⚡ АНАЛИЗ ПРОСТРАНСТВА ВСЕХ 7 ФУНДАМЕНТАЛЬНЫХ КОНСТАНТ:")
    print("   α, m_p, m_e, G, c, ħ, ε₀")
    
    hv = HyperVolume7D()
    
    results = hv.generate_7d_adaptive(
        alpha_range=(1/400, 1/15),
        m_p_range=(0.1, 5.0),
        m_e_range=(0.1, 5.0),
        G_range=(0.05, 10.0),
        c_range=(0.2, 3.0),
        hbar_range=(0.2, 3.0),
        epsilon_0_range=(0.1, 5.0),
        coarse_points=3,
        zoom_points=4,
        zoom_fraction=0.25,
        max_refinements=2
    )
    
    volume = hv.calculate_7d_volume(threshold=0.6)
    
    our_universe = UniverseParameters(name="🌍 Наша Вселенная")
    our_analyzer = UniverseAnalyzer(our_universe)
    _, our_score, _ = our_analyzer.calculate_habitability_index()
    
    print(f"\n🌍 НАША ВСЕЛЕННАЯ: score = {our_score:.3f}")
    print(f"🌟 ОПТИМУМ (7D): score = {results['best_score']:.3f}")
    if volume:
        print(f"📊 Доля пригодного 7D пространства: {volume['fraction']*100:.2f}%")
    
    print("\n" + "="*90)
    print("🎉 7D АНАЛИЗ ЗАВЕРШЕН!")
    print("="*90)


if __name__ == "__main__":
    main()
