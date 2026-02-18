#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D Ландшафт пригодности вселенных
Исследуем пространство параметров: α, m_p, m_e
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Импортируем из основного модуля
try:
    from multiverse_tester import (
        UniverseParameters, UniversalConstants, UniverseAnalyzer,
        AtomicPhysics, NuclearPhysics, StellarNucleosynthesis,
        HabitabilityIndex
    )
except ImportError:
    print("⚠️ Создаем временные классы для демонстрации...")
    
    # Минимальная реализация для демонстрации
    class HabitabilityIndex:
        DEAD, HOSTILE, MARGINAL, HABITABLE, OPTIMAL = range(5)
    
    class UniversalConstants:
        def __init__(self):
            self.m_e = 9.10938356e-31
            self.m_p = 1.6726219e-27
            self.e = 1.60217662e-19
            self.hbar = 1.0545718e-34
            self.c = 299792458.0
            self.epsilon_0 = 8.8541878128e-12
            self.G = 6.67430e-11
            self.k_B = 1.380649e-23
    
    class UniverseParameters:
        def __init__(self, name="Test", alpha=None, m_p=None, m_e=None):
            self.name = name
            self.const = UniversalConstants()
            self.alpha = alpha if alpha else 1/137.036
            self.m_p = m_p if m_p else self.const.m_p
            self.m_e = m_e if m_e else self.const.m_e
            self.e = math.sqrt(self.alpha * 4 * math.pi * self.const.epsilon_0 * 
                              self.const.hbar * self.const.c)
        
        def __repr__(self):
            return f"{self.name}: α={self.alpha:.6f}"
    
    class UniverseAnalyzer:
        def __init__(self, universe):
            self.u = universe
            
        def calculate_habitability_index(self):
            # Упрощенная модель для демонстрации
            score = 0.0
            
            # 1. Атомная стабильность
            a0_ratio = (self.u.alpha / 0.007297) * (self.u.m_e / 9.11e-31)**0.5
            if 0.5 < a0_ratio < 2:
                score += 0.25
            
            # 2. Химия
            if 0.003 < self.u.alpha < 0.02:
                score += 0.25
            
            # 3. Ядерная стабильность
            binding = 8.5 * (self.u.alpha / 0.007297)**(-0.5) * (self.u.m_p / 1.67e-27)
            if 4 < binding < 12:
                score += 0.25
            
            # 4. Звездный синтез
            triple_alpha = math.exp(-abs(self.u.alpha - 0.007297)/0.005)
            score += 0.25 * triple_alpha
            
            return None, score, {}


class Landscape3D:
    """
    3D ландшафт пригодности вселенных
    """
    
    def __init__(self):
        self.const = UniversalConstants()
        self.results = {}
        
    def generate_3d_grid(self, 
                         alpha_range: Tuple[float, float] = (1/300, 1/30),
                         m_p_range: Tuple[float, float] = (0.5, 2.0),
                         m_e_range: Tuple[float, float] = (0.5, 2.0),
                         points: int = 30) -> Dict:
        """
        Генерирует 3D сетку пригодности
        
        Аргументы:
            alpha_range: (min, max) для α
            m_p_range: (min, max) для массы протона (в единицах нашей)
            m_e_range: (min, max) для массы электрона (в единицах нашей)
            points: количество точек по каждому измерению
        """
        print(f"\n🔮 ГЕНЕРАЦИЯ 3D ЛАНДШАФТА {points}×{points}×{points}")
        print(f"   α: [{alpha_range[0]:.4f}, {alpha_range[1]:.4f}]")
        print(f"   m_p/m_p₀: [{m_p_range[0]:.2f}, {m_p_range[1]:.2f}]")
        print(f"   m_e/m_e₀: [{m_e_range[0]:.2f}, {m_e_range[1]:.2f}]")
        
        # Создаем сетку
        alphas = np.linspace(alpha_range[0], alpha_range[1], points)
        m_p_ratios = np.linspace(m_p_range[0], m_p_range[1], points)
        m_e_ratios = np.linspace(m_e_range[0], m_e_range[1], points)
        
        # Создаем 3D массив для результатов
        score_3d = np.zeros((points, points, points))
        category_3d = np.zeros((points, points, points))
        
        total_points = points ** 3
        count = 0
        
        # Полный перебор
        for i, alpha in enumerate(alphas):
            for j, m_p_ratio in enumerate(m_p_ratios):
                for k, m_e_ratio in enumerate(m_e_ratios):
                    try:
                        u = UniverseParameters(
                            alpha=alpha,
                            m_p=m_p_ratio * self.const.m_p,
                            m_e=m_e_ratio * self.const.m_e
                        )
                        analyzer = UniverseAnalyzer(u)
                        _, score, _ = analyzer.calculate_habitability_index()
                        
                        score_3d[i, j, k] = score
                        
                        # Категория
                        if score > 0.8:
                            category_3d[i, j, k] = 4  # OPTIMAL
                        elif score > 0.6:
                            category_3d[i, j, k] = 3  # HABITABLE
                        elif score > 0.3:
                            category_3d[i, j, k] = 2  # MARGINAL
                        elif score > 0.1:
                            category_3d[i, j, k] = 1  # HOSTILE
                        else:
                            category_3d[i, j, k] = 0  # DEAD
                            
                    except Exception as e:
                        score_3d[i, j, k] = 0
                        category_3d[i, j, k] = 0
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f"   Прогресс: {count}/{total_points} ({count/total_points*100:.1f}%)")
        
        # Находим глобальный максимум
        max_idx = np.unravel_index(np.argmax(score_3d), score_3d.shape)
        best_alpha = alphas[max_idx[0]]
        best_m_p = m_p_ratios[max_idx[1]]
        best_m_e = m_e_ratios[max_idx[2]]
        best_score = score_3d[max_idx]
        
        print(f"\n✅ ГЛОБАЛЬНЫЙ ОПТИМУМ:")
        print(f"   α = {best_alpha:.6f}")
        print(f"   m_p/m_p₀ = {best_m_p:.3f}")
        print(f"   m_e/m_e₀ = {best_m_e:.3f}")
        print(f"   Индекс пригодности = {best_score:.3f}")
        
        self.results = {
            'alphas': alphas,
            'm_p_ratios': m_p_ratios,
            'm_e_ratios': m_e_ratios,
            'score_3d': score_3d,
            'category_3d': category_3d,
            'best_alpha': best_alpha,
            'best_m_p': best_m_p,
            'best_m_e': best_m_e,
            'best_score': best_score
        }
        
        return self.results
    
    def find_habitable_volume(self, threshold: float = 0.6) -> Dict:
        """
        Находит объем пространства параметров, пригодный для жизни
        """
        if not self.results:
            print("❌ Сначала сгенерируйте 3D сетку!")
            return {}
        
        score = self.results['score_3d']
        habitable_mask = score > threshold
        
        # Объем в единицах сетки
        voxel_count = np.sum(habitable_mask)
        total_voxels = score.size
        volume_fraction = voxel_count / total_voxels
        
        # Координаты пригодных точек
        indices = np.where(habitable_mask)
        
        # Диапазоны параметров
        alphas = self.results['alphas'][indices[0]]
        m_p_ratios = self.results['m_p_ratios'][indices[1]]
        m_e_ratios = self.results['m_e_ratios'][indices[2]]
        
        ranges = {
            'alpha': (alphas.min(), alphas.max()),
            'm_p': (m_p_ratios.min(), m_p_ratios.max()),
            'm_e': (m_e_ratios.min(), m_e_ratios.max())
        }
        
        print(f"\n📊 ПРИГОДНЫЙ ОБЪЕМ (score > {threshold}):")
        print(f"   Доля пространства: {volume_fraction*100:.2f}%")
        print(f"   Количество точек: {voxel_count}/{total_voxels}")
        print(f"\n   Диапазоны:")
        print(f"   α: [{ranges['alpha'][0]:.4f}, {ranges['alpha'][1]:.4f}]")
        print(f"   m_p/m_p₀: [{ranges['m_p'][0]:.2f}, {ranges['m_p'][1]:.2f}]")
        print(f"   m_e/m_e₀: [{ranges['m_e'][0]:.2f}, {ranges['m_e'][1]:.2f}]")
        
        return {
            'fraction': volume_fraction,
            'voxel_count': voxel_count,
            'ranges': ranges,
            'mask': habitable_mask
        }


class LandscapeVisualizer3D:
    """
    Визуализация 3D ландшафта
    """
    
    def __init__(self, landscape: Landscape3D):
        self.land = landscape
        self.results = landscape.results
        
    def plot_3d_scatter(self, threshold: float = 0.6, 
                        figsize: Tuple[int, int] = (14, 10)):
        """
        3D scatter plot пригодных вселенных
        """
        if not self.results:
            print("❌ Нет данных для визуализации")
            return
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Получаем данные
        alphas = self.results['alphas']
        m_p_ratios = self.results['m_p_ratios']
        m_e_ratios = self.results['m_e_ratios']
        scores = self.results['score_3d']
        
        # Создаем сетку координат
        X, Y, Z = np.meshgrid(alphas, m_p_ratios, m_e_ratios, indexing='ij')
        
        # Маска для пригодных точек
        mask = scores > threshold
        
        # Цвета по пригодности
        colors = scores[mask]
        
        # Отображаем только пригодные точки
        ax.scatter(X[mask], Y[mask], Z[mask], 
                  c=colors, cmap='RdYlGn', s=20, alpha=0.6, vmin=0, vmax=1)
        
        # Отмечаем нашу Вселенную
        ax.scatter([1/137.036], [1.0], [1.0], 
                  c='red', s=200, marker='*', label='🌍 Наша Вселенная')
        
        # Отмечаем глобальный оптимум
        ax.scatter([self.results['best_alpha']], 
                  [self.results['best_m_p']], 
                  [self.results['best_m_e']],
                  c='blue', s=200, marker='*', label='🌟 Глобальный оптимум')
        
        ax.set_xlabel('α', fontsize=12)
        ax.set_ylabel('m_p / m_p₀', fontsize=12)
        ax.set_zlabel('m_e / m_e₀', fontsize=12)
        ax.set_title(f'3D Ландшафт пригодности (score > {threshold})', fontsize=14)
        
        # Добавляем colorbar
        mappable = cm.ScalarMappable(cmap='RdYlGn')
        mappable.set_array(scores[mask])
        plt.colorbar(mappable, ax=ax, label='Индекс пригодности', shrink=0.5)
        
        ax.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_slices(self, slice_values: Dict[str, float], 
                   figsize: Tuple[int, int] = (15, 10)):
        """
        Строит срезы 3D пространства
        
        Аргументы:
            slice_values: {'alpha': значение, 'm_p': значение, 'm_e': значение}
        """
        if not self.results:
            print("❌ Нет данных для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        alphas = self.results['alphas']
        m_p_ratios = self.results['m_p_ratios']
        m_e_ratios = self.results['m_e_ratios']
        scores = self.results['score_3d']
        
        # 1. Срез при фиксированном α
        if 'alpha' in slice_values:
            ax = axes[0, 0]
            alpha_idx = np.argmin(np.abs(alphas - slice_values['alpha']))
            slice_2d = scores[alpha_idx, :, :]
            
            im = ax.imshow(slice_2d.T, origin='lower', 
                          extent=[m_p_ratios[0], m_p_ratios[-1], 
                                 m_e_ratios[0], m_e_ratios[-1]],
                          aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Пригодность')
            ax.set_xlabel('m_p / m_p₀')
            ax.set_ylabel('m_e / m_e₀')
            ax.set_title(f'Срез при α = {slice_values["alpha"]:.4f}')
            
            # Наша Вселенная
            if abs(slice_values['alpha'] - 1/137.036) < 0.001:
                ax.plot(1.0, 1.0, 'r*', markersize=15, label='🌍')
        
        # 2. Срез при фиксированном m_p
        if 'm_p' in slice_values:
            ax = axes[0, 1]
            m_p_idx = np.argmin(np.abs(m_p_ratios - slice_values['m_p']))
            slice_2d = scores[:, m_p_idx, :]
            
            im = ax.imshow(slice_2d.T, origin='lower', 
                          extent=[alphas[0], alphas[-1], 
                                 m_e_ratios[0], m_e_ratios[-1]],
                          aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Пригодность')
            ax.set_xlabel('α')
            ax.set_ylabel('m_e / m_e₀')
            ax.set_title(f'Срез при m_p/m_p₀ = {slice_values["m_p"]:.2f}')
            
            if abs(slice_values['m_p'] - 1.0) < 0.01:
                ax.plot(1/137.036, 1.0, 'r*', markersize=15, label='🌍')
        
        # 3. Срез при фиксированном m_e
        if 'm_e' in slice_values:
            ax = axes[1, 0]
            m_e_idx = np.argmin(np.abs(m_e_ratios - slice_values['m_e']))
            slice_2d = scores[:, :, m_e_idx]
            
            im = ax.imshow(slice_2d.T, origin='lower', 
                          extent=[alphas[0], alphas[-1], 
                                 m_p_ratios[0], m_p_ratios[-1]],
                          aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Пригодность')
            ax.set_xlabel('α')
            ax.set_ylabel('m_p / m_p₀')
            ax.set_title(f'Срез при m_e/m_e₀ = {slice_values["m_e"]:.2f}')
            
            if abs(slice_values['m_e'] - 1.0) < 0.01:
                ax.plot(1/137.036, 1.0, 'r*', markersize=15, label='🌍')
        
        # 4. Проекция максимальной пригодности
        ax = axes[1, 1]
        max_projection = np.max(scores, axis=2)  # максимум по m_e
        
        im = ax.imshow(max_projection.T, origin='lower', 
                      extent=[alphas[0], alphas[-1], 
                             m_p_ratios[0], m_p_ratios[-1]],
                      aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Макс. пригодность')
        ax.set_xlabel('α')
        ax.set_ylabel('m_p / m_p₀')
        ax.set_title('Максимум по всем m_e')
        
        # Наша Вселенная
        ax.plot(1/137.036, 1.0, 'r*', markersize=15, label='🌍 Наша')
        
        # Глобальный оптимум
        ax.plot(self.results['best_alpha'], self.results['best_m_p'], 
               'b*', markersize=15, label='🌟 Оптимум')
        
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_isosurface(self, threshold: float = 0.6, 
                       figsize: Tuple[int, int] = (12, 8)):
        """
        Изоповерхность постоянной пригодности
        """
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from skimage import measure
        
        if not self.results:
            print("❌ Нет данных для визуализации")
            return
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Получаем данные
        scores = self.results['score_3d']
        
        # Создаем изоповерхность
        verts, faces, _, _ = measure.marching_cubes(scores, threshold)
        
        # Масштабируем координаты
        alphas = self.results['alphas']
        m_p_ratios = self.results['m_p_ratios']
        m_e_ratios = self.results['m_e_ratios']
        
        verts[:, 0] = alphas[0] + verts[:, 0] * (alphas[-1] - alphas[0]) / scores.shape[0]
        verts[:, 1] = m_p_ratios[0] + verts[:, 1] * (m_p_ratios[-1] - m_p_ratios[0]) / scores.shape[1]
        verts[:, 2] = m_e_ratios[0] + verts[:, 2] * (m_e_ratios[-1] - m_e_ratios[0]) / scores.shape[2]
        
        # Отображаем
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                       cmap='RdYlGn', alpha=0.8)
        
        # Наша Вселенная
        ax.scatter([1/137.036], [1.0], [1.0], 
                  c='red', s=200, marker='*', label='🌍 Наша')
        
        ax.set_xlabel('α', fontsize=12)
        ax.set_ylabel('m_p / m_p₀', fontsize=12)
        ax.set_zlabel('m_e / m_e₀', fontsize=12)
        ax.set_title(f'Изоповерхность пригодности (threshold = {threshold})', fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.show()


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    """Запуск 3D анализа"""
    
    print("="*70)
    print("🌌 3D ЛАНДШАФТ ПРИГОДНОСТИ ВСЕЛЕННЫХ v1.0")
    print("="*70)
    
    # Создаем ландшафт
    landscape = Landscape3D()
    
    # Генерируем 3D сетку (уменьшенное разрешение для скорости)
    # Для полного анализа можно увеличить points до 50-100
    results = landscape.generate_3d_grid(
        alpha_range=(1/300, 1/30),
        m_p_range=(0.5, 2.0),
        m_e_range=(0.5, 2.0),
        points=30  # 30×30×30 = 27,000 точек
    )
    
    # Создаем визуализатор
    viz = LandscapeVisualizer3D(landscape)
    
    # 1. 3D scatter plot пригодных вселенных
    print("\n📊 ВИЗУАЛИЗАЦИЯ 1: 3D scatter plot")
    viz.plot_3d_scatter(threshold=0.6)
    
    # 2. Срезы через нашу Вселенную
    print("\n📊 ВИЗУАЛИЗАЦИЯ 2: Срезы через нашу Вселенную")
    viz.plot_slices({
        'alpha': 1/137.036,
        'm_p': 1.0,
        'm_e': 1.0
    })
    
    # 3. Срезы через глобальный оптимум
    print("\n📊 ВИЗУАЛИЗАЦИЯ 3: Срезы через глобальный оптимум")
    viz.plot_slices({
        'alpha': results['best_alpha'],
        'm_p': results['best_m_p'],
        'm_e': results['best_m_e']
    })
    
    # 4. Объем пригодного пространства
    print("\n📊 ВИЗУАЛИЗАЦИЯ 4: Анализ объема")
    volume = landscape.find_habitable_volume(threshold=0.6)
    
    # 5. Изоповерхность (требуется scikit-image)
    try:
        print("\n📊 ВИЗУАЛИЗАЦИЯ 5: Изоповерхность")
        viz.plot_isosurface(threshold=0.6)
    except ImportError:
        print("⚠️ Для изоповерхности требуется scikit-image:")
        print("   pip install scikit-image")
    
    # 6. ИТОГОВЫЙ ОТЧЕТ
    print("\n" + "="*70)
    print("📈 ИТОГОВЫЙ 3D АНАЛИЗ")
    print("="*70)
    
    # Наша Вселенная
    our_analyzer = UniverseAnalyzer(UniverseParameters())
    _, our_score, _ = our_analyzer.calculate_habitability_index()
    
    print(f"\n🌍 НАША ВСЕЛЕННАЯ:")
    print(f"   α = {1/137.036:.6f}")
    print(f"   m_p/m_p₀ = 1.000")
    print(f"   m_e/m_e₀ = 1.000")
    print(f"   Индекс пригодности = {our_score:.3f}")
    
    print(f"\n🌟 ГЛОБАЛЬНЫЙ ОПТИМУМ:")
    print(f"   α = {results['best_alpha']:.6f}")
    print(f"   m_p/m_p₀ = {results['best_m_p']:.3f}")
    print(f"   m_e/m_e₀ = {results['best_m_e']:.3f}")
    print(f"   Индекс пригодности = {results['best_score']:.3f}")
    
    if volume:
        print(f"\n📊 ПРИГОДНЫЙ ОБЪЕМ (score > 0.6):")
        print(f"   Доля пространства: {volume['fraction']*100:.2f}%")
        print(f"   Диапазон α: [{volume['ranges']['alpha'][0]:.4f}, {volume['ranges']['alpha'][1]:.4f}]")
        print(f"   Диапазон m_p: [{volume['ranges']['m_p'][0]:.2f}, {volume['ranges']['m_p'][1]:.2f}]")
        print(f"   Диапазон m_e: [{volume['ranges']['m_e'][0]:.2f}, {volume['ranges']['m_e'][1]:.2f}]")
    
    # 7. Проекции
    print(f"\n📈 ПРОЕКЦИИ НА ПЛОСКОСТИ:")
    
    # Проекция на плоскость α-m_p (максимум по m_e)
    max_over_me = np.max(results['score_3d'], axis=2)
    best_alpha_idx, best_mp_idx = np.unravel_index(np.argmax(max_over_me), max_over_me.shape)
    
    print(f"\n   α-m_p плоскость (макс по m_e):")
    print(f"   Оптимум: α={results['alphas'][best_alpha_idx]:.4f}, m_p={results['m_p_ratios'][best_mp_idx]:.2f}")
    
    # Проекция на плоскость α-m_e (максимум по m_p)
    max_over_mp = np.max(results['score_3d'], axis=1)
    best_alpha_idx2, best_me_idx = np.unravel_index(np.argmax(max_over_mp), max_over_mp.shape)
    
    print(f"\n   α-m_e плоскость (макс по m_p):")
    print(f"   Оптимум: α={results['alphas'][best_alpha_idx2]:.4f}, m_e={results['m_e_ratios'][best_me_idx]:.2f}")
    
    # Проекция на плоскость m_p-m_e (максимум по α)
    max_over_alpha = np.max(results['score_3d'], axis=0)
    best_mp_idx2, best_me_idx2 = np.unravel_index(np.argmax(max_over_alpha), max_over_alpha.shape)
    
    print(f"\n   m_p-m_e плоскость (макс по α):")
    print(f"   Оптимум: m_p={results['m_p_ratios'][best_mp_idx2]:.2f}, m_e={results['m_e_ratios'][best_me_idx2]:.2f}")
    
    print("\n" + "="*70)
    print("🎉 3D АНАЛИЗ ЗАВЕРШЕН!")
    print("="*70)


if __name__ == "__main__":
    main()
