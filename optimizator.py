import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, differential_evolution
from dataclasses import dataclass
from typing import Tuple, List, Dict, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

from main import UniverseParameters, UniverseAnalyzer, UniversalConstants

# ==================== ОПТИМИЗАТОР ВСЕЛЕННОЙ ====================

class UniverseOptimizer:
    """
    Оптимизатор параметров Вселенной для максимальной пригодности жизни
    """
    
    def __init__(self):
        self.best_universes = []
        self.optimization_history = []
        
    def objective_function(self, alpha: float, m_p_ratio: float = 1.0, 
                          verbose: bool = False) -> float:
        """
        Целевая функция для оптимизации (чем меньше, тем лучше)
        
        Возвращает: 1 - индекс пригодности (чем меньше, тем ближе к оптимуму)
        """
        try:
            # Создаем вселенную с заданными параметрами
            u = UniverseParameters(
                name=f"Test α={alpha:.6f}, m_p/m_p₀={m_p_ratio:.3f}",
                alpha=alpha,
                m_p=m_p_ratio * UniversalConstants().m_p
            )
            
            # Анализируем
            analyzer = UniverseAnalyzer(u)
            index, score, metrics = analyzer.calculate_habitability_index()
            
            # Сохраняем в историю
            self.optimization_history.append({
                'alpha': alpha,
                'm_p_ratio': m_p_ratio,
                'score': score,
                'index': index.name,
                'metrics': metrics
            })
            
            if verbose:
                print(f"   α={alpha:.6f}, m_p/m_p₀={m_p_ratio:.3f} → score={score:.3f} ({index.name})")
            
            # Целевая функция: минимизируем 1-score
            return 1.0 - score
            
        except Exception as e:
            if verbose:
                print(f"   Ошибка при α={alpha}: {e}")
            return 1.0  # максимальная "непригодность"
    
    def optimize_alpha(self, bounds: Tuple[float, float] = (1/300, 1/30), 
                      method: str = 'brent', verbose: bool = True) -> Dict:
        """
        Оптимизирует только α (при фиксированной массе протона)
        
        Аргументы:
            bounds: (min, max) диапазон α
            method: 'brent' или 'golden'
            verbose: выводить детали
        """
        print(f"\n🎯 ОПТИМИЗАЦИЯ α В ДИАПАЗОНЕ [{bounds[0]:.6f}, {bounds[1]:.6f}]")
        
        # Очищаем историю
        self.optimization_history = []
        
        if method == 'brent':
            # Метод Брента (быстрый и точный)
            result = minimize_scalar(
                lambda x: self.objective_function(x, verbose=False),
                bounds=bounds,
                method='bounded',
                options={'xatol': 1e-6}
            )
            optimal_alpha = result.x
            min_objective = result.fun
        else:
            # Золотое сечение (медленнее, но надежнее)
            result = minimize_scalar(
                lambda x: self.objective_function(x, verbose=False),
                bounds=bounds,
                method='golden',
                options={'xtol': 1e-6}
            )
            optimal_alpha = result.x
            min_objective = result.fun
        
        # Финальная оценка
        final_score = 1.0 - min_objective
        print(f"\n✅ Оптимальная α = {optimal_alpha:.6f}")
        print(f"   Индекс пригодности = {final_score:.3f}")
        
        # Анализируем оптимальную вселенную детально
        opt_universe = UniverseParameters(
            name=f"🌌 Оптимальная α={optimal_alpha:.6f}",
            alpha=optimal_alpha
        )
        analyzer = UniverseAnalyzer(opt_universe)
        index, score, metrics = analyzer.calculate_habitability_index()
        
        return {
            'alpha': optimal_alpha,
            'score': score,
            'index': index.name,
            'metrics': metrics,
            'history': self.optimization_history
        }
    
    def optimize_2d(self, alpha_bounds: Tuple[float, float] = (1/300, 1/30),
                   m_p_bounds: Tuple[float, float] = (0.5, 2.0),
                   popsize: int = 50, maxiter: int = 100,
                   verbose: bool = True) -> Dict:
        """
        Двумерная оптимизация (α и масса протона)
        Использует дифференциальную эволюцию
        
        Аргументы:
            alpha_bounds: (min, max) α
            m_p_bounds: (min, max) масса протона относительно нашей
            popsize: размер популяции
            maxiter: максимальное число итераций
        """
        print(f"\n🎯 2D ОПТИМИЗАЦИЯ:")
        print(f"   α: [{alpha_bounds[0]:.6f}, {alpha_bounds[1]:.6f}]")
        print(f"   m_p/m_p₀: [{m_p_bounds[0]:.3f}, {m_p_bounds[1]:.3f}]")
        
        # Очищаем историю
        self.optimization_history = []
        
        # Границы для оптимизации
        bounds = [alpha_bounds, m_p_bounds]
        
        # Целевая функция для 2D
        def objective_2d(x):
            alpha, m_p_ratio = x
            return self.objective_function(alpha, m_p_ratio, verbose=False)
        
        # Дифференциальная эволюция
        result = differential_evolution(
            objective_2d,
            bounds,
            strategy='best1bin',
            popsize=popsize,
            maxiter=maxiter,
            tol=1e-6,
            updating='deferred',
            workers=1
        )
        
        optimal_alpha, optimal_m_p = result.x
        min_objective = result.fun
        final_score = 1.0 - min_objective
        
        print(f"\n✅ ОПТИМАЛЬНАЯ ВСЕЛЕННАЯ:")
        print(f"   α = {optimal_alpha:.6f}")
        print(f"   m_p/m_p₀ = {optimal_m_p:.3f}")
        print(f"   Индекс пригодности = {final_score:.3f}")
        
        # Анализируем оптимальную вселенную
        opt_universe = UniverseParameters(
            name=f"🌌 Оптимальная α={optimal_alpha:.6f}, m_p/m_p₀={optimal_m_p:.3f}",
            alpha=optimal_alpha,
            m_p=optimal_m_p * UniversalConstants().m_p
        )
        analyzer = UniverseAnalyzer(opt_universe)
        index, score, metrics = analyzer.calculate_habitability_index()
        
        # Детальный нуклеосинтез
        nucleo = analyzer.stellar.complete_nucleosynthesis_analysis()
        
        return {
            'alpha': optimal_alpha,
            'm_p_ratio': optimal_m_p,
            'score': score,
            'index': index.name,
            'metrics': metrics,
            'nucleosynthesis': nucleo,
            'history': self.optimization_history,
            'success': result.success
        }
    
    def grid_search(self, alpha_points: int = 50, m_p_points: int = 30,
                   alpha_range: Tuple[float, float] = (1/300, 1/30),
                   m_p_range: Tuple[float, float] = (0.5, 2.0)) -> Dict:
        """
        Полный перебор по сетке для визуализации ландшафта пригодности
        
        Аргументы:
            alpha_points: количество точек по α
            m_p_points: количество точек по массе протона
        """
        print(f"\n🔍 ПОЛНЫЙ ПЕРЕБОР ПО СЕТКЕ {alpha_points}×{m_p_points}...")
        
        alphas = np.linspace(alpha_range[0], alpha_range[1], alpha_points)
        m_p_ratios = np.linspace(m_p_range[0], m_p_range[1], m_p_points)
        
        score_map = np.zeros((alpha_points, m_p_points))
        category_map = np.zeros((alpha_points, m_p_points))
        
        total_points = alpha_points * m_p_points
        count = 0
        
        for i, alpha in enumerate(alphas):
            for j, m_p_ratio in enumerate(m_p_ratios):
                try:
                    u = UniverseParameters(
                        alpha=alpha,
                        m_p=m_p_ratio * UniversalConstants().m_p
                    )
                    analyzer = UniverseAnalyzer(u)
                    _, score, _ = analyzer.calculate_habitability_index()
                    
                    score_map[i, j] = score
                    
                    # Категория для цветовой карты
                    if score > 0.8:
                        category_map[i, j] = 4  # OPTIMAL
                    elif score > 0.6:
                        category_map[i, j] = 3  # HABITABLE
                    elif score > 0.3:
                        category_map[i, j] = 2  # MARGINAL
                    elif score > 0.1:
                        category_map[i, j] = 1  # HOSTILE
                    else:
                        category_map[i, j] = 0  # DEAD
                        
                except Exception as e:
                    score_map[i, j] = 0
                    category_map[i, j] = 0
                
                count += 1
                if count % 100 == 0:
                    print(f"   Прогресс: {count}/{total_points} ({count/total_points*100:.1f}%)")
        
        # Находим максимум
        max_idx = np.unravel_index(np.argmax(score_map), score_map.shape)
        best_alpha = alphas[max_idx[0]]
        best_m_p = m_p_ratios[max_idx[1]]
        best_score = score_map[max_idx]
        
        print(f"\n✅ ЛУЧШАЯ ПО СЕТКЕ:")
        print(f"   α = {best_alpha:.6f}")
        print(f"   m_p/m_p₀ = {best_m_p:.3f}")
        print(f"   Индекс пригодности = {best_score:.3f}")
        
        return {
            'alphas': alphas,
            'm_p_ratios': m_p_ratios,
            'score_map': score_map,
            'category_map': category_map,
            'best_alpha': best_alpha,
            'best_m_p': best_m_p,
            'best_score': best_score
        }


# ==================== ВИЗУАЛИЗАЦИЯ ОПТИМИЗАЦИИ ====================

class OptimizationVisualizer:
    """Визуализация результатов оптимизации"""
    
    def __init__(self, optimizer: UniverseOptimizer):
        self.opt = optimizer
        
    def plot_optimization_1d(self, result: Dict, figsize: Tuple[int, int] = (12, 8)):
        """Строит график 1D оптимизации"""
        
        history = result['history']
        alphas = [h['alpha'] for h in history]
        scores = [h['score'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Траектория оптимизации
        ax = axes[0, 0]
        ax.plot(alphas, scores, 'b.-', alpha=0.5, label='Пробные точки')
        ax.axvline(x=result['alpha'], color='r', linestyle='--', linewidth=2, 
                  label=f"Оптимум α={result['alpha']:.6f}")
        ax.axhline(y=result['score'], color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('α')
        ax.set_ylabel('Индекс пригодности')
        ax.set_title('Траектория оптимизации α')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Распределение метрик в оптимуме
        ax = axes[0, 1]
        metrics = result['metrics']
        names = list(metrics.keys())
        values = list(metrics.values())
        colors = ['green' if v > 0.8 else 'yellow' if v > 0.5 else 'red' for v in values]
        ax.bar(names, values, color=colors, alpha=0.7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Значение')
        ax.set_title('Метрики в оптимальной вселенной')
        ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Порог пригодности')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Гистограмма пробных точек
        ax = axes[1, 0]
        ax.hist(scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=result['score'], color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Индекс пригодности')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение пробных точек')
        ax.grid(True, alpha=0.3)
        
        # 4. Сходимость
        ax = axes[1, 1]
        best_so_far = np.maximum.accumulate(scores)
        ax.plot(best_so_far, 'g-', linewidth=2, label='Лучший найденный')
        ax.plot(scores, 'b.', alpha=0.3, label='Все точки')
        ax.set_xlabel('Номер итерации')
        ax.set_ylabel('Индекс пригодности')
        ax.set_title('Сходимость оптимизации')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_landscape_2d(self, grid_result: Dict, figsize: Tuple[int, int] = (14, 6)):
        """Строит ландшафт пригодности 2D"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        X, Y = np.meshgrid(grid_result['alphas'], grid_result['m_p_ratios'], indexing='ij')
        
        # 1. Тепловая карта пригодности
        im1 = ax1.pcolormesh(X, Y, grid_result['score_map'], 
                            cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
        plt.colorbar(im1, ax=ax1, label='Индекс пригодности')
        
        # Отмечаем нашу Вселенную
        ax1.plot(1/137.036, 1.0, 'r*', markersize=15, label='Наша Вселенная')
        
        # Отмечаем оптимальную
        ax1.plot(grid_result['best_alpha'], grid_result['best_m_p'], 
                'b*', markersize=15, label='Оптимальная')
        
        ax1.set_xlabel('α')
        ax1.set_ylabel('m_p / m_p₀')
        ax1.set_title('Ландшафт пригодности для жизни')
        ax1.legend()
        
        # 2. Карта категорий
        im2 = ax2.pcolormesh(X, Y, grid_result['category_map'],
                            cmap='RdYlGn', vmin=0, vmax=4, shading='auto')
        cbar = plt.colorbar(im2, ax=ax2, label='Категория')
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.set_ticklabels(['DEAD', 'HOSTILE', 'MARGINAL', 'HABITABLE', 'OPTIMAL'])
        
        ax2.plot(1/137.036, 1.0, 'r*', markersize=15, label='Наша Вселенная')
        ax2.plot(grid_result['best_alpha'], grid_result['best_m_p'], 
                'b*', markersize=15, label='Оптимальная')
        
        ax2.set_xlabel('α')
        ax2.set_ylabel('m_p / m_p₀')
        ax2.set_title('Категории вселенных')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, universes: List[UniverseParameters], 
                        names: List[str], figsize: Tuple[int, int] = (15, 8)):
        """Сравнивает несколько вселенных"""
        
        n = len(universes)
        fig, axes = plt.subplots(2, n, figsize=figsize)
        
        for i, (u, name) in enumerate(zip(universes, names)):
            analyzer = UniverseAnalyzer(u)
            index, score, metrics = analyzer.calculate_habitability_index()
            
            # Верхний график: метрики
            ax = axes[0, i] if n > 1 else axes[0]
            names_m = list(metrics.keys())
            values = list(metrics.values())
            colors = ['green' if v > 0.8 else 'yellow' if v > 0.5 else 'red' for v in values]
            
            ax.bar(names_m, values, color=colors, alpha=0.7)
            ax.set_ylim(0, 1.1)
            ax.set_title(f'{name}\n(score={score:.3f}, {index.name})')
            ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Нижний график: нуклеосинтез
            ax = axes[1, i] if n > 1 else axes[1]
            nucleo = analyzer.stellar.complete_nucleosynthesis_analysis()
            
            # Основные показатели нуклеосинтеза
            synth_data = {
                'pp': nucleo['pp_chain']['rate_relative'],
                'CNO': nucleo['cno_cycle']['rate_relative'],
                '3α': nucleo['triple_alpha']['rate_relative'],
                'α-proc': np.mean([r['relative_yield'] for r in nucleo['alpha_process'][:5]]),
                's-proc': 1.0 if 'медленный' in nucleo['s_process']['path'] else 0.5,
                'r-proc': nucleo['r_process']['transuranic_elements'] / 10
            }
            
            names_s = list(synth_data.keys())
            values_s = list(synth_data.values())
            ax.bar(names_s, values_s, color='blue', alpha=0.6)
            ax.set_ylim(0, max(1.5, max(values_s)))
            ax.set_title('Эффективность нуклеосинтеза')
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Наш уровень')
        
        plt.tight_layout()
        plt.show()


# ==================== ЗАПУСК ОПТИМИЗАЦИИ ====================

if __name__ == "__main__":
    
    print("="*60)
    print("🚀 ОПТИМИЗАТОР ВСЕЛЕННЫХ v1.0")
    print("="*60)
    
    # Создаем оптимизатор
    optimizer = UniverseOptimizer()
    visualizer = OptimizationVisualizer(optimizer)
    
    # 1. Оптимизация только α
    print("\n" + "="*60)
    print("1️⃣ ОПТИМИЗАЦИЯ α")
    print("="*60)
    
    opt_result = optimizer.optimize_alpha(
        bounds=(1/300, 1/30),
        method='brent',
        verbose=True
    )
    
    visualizer.plot_optimization_1d(opt_result)
    
    # 2. Полный перебор по сетке (для ландшафта)
    print("\n" + "="*60)
    print("2️⃣ ПОЛНЫЙ ПЕРЕБОР ПО СЕТКЕ")
    print("="*60)
    
    grid = optimizer.grid_search(
        alpha_points=100,
        m_p_points=50,
        alpha_range=(1/300, 1/30),
        m_p_range=(0.5, 2.0)
    )
    
    visualizer.plot_landscape_2d(grid)
    
    # 3. Двумерная оптимизация
    print("\n" + "="*60)
    print("3️⃣ 2D ОПТИМИЗАЦИЯ (α и m_p)")
    print("="*60)
    
    opt_2d = optimizer.optimize_2d(
        alpha_bounds=(1/300, 1/30),
        m_p_bounds=(0.5, 2.0),
        popsize=30,
        maxiter=50,
        verbose=True
    )
    
    # 4. Сравнение вселенных
    print("\n" + "="*60)
    print("4️⃣ СРАВНЕНИЕ ВСЕЛЕННЫХ")
    print("="*60)
    
    universes = [
        UniverseParameters(name="🌍 Наша", alpha=1/137.036),
        UniverseParameters(name=f"✨ Оптимальная α={opt_result['alpha']:.4f}", 
                          alpha=opt_result['alpha']),
        UniverseParameters(name=f"🌟 2D Оптимум α={opt_2d['alpha']:.4f}, m_p/m_p₀={opt_2d['m_p_ratio']:.2f}", 
                          alpha=opt_2d['alpha'], 
                          m_p=opt_2d['m_p_ratio'] * UniversalConstants().m_p),
        UniverseParameters(name="💀 Экстремальная", alpha=1/50, m_p=2 * UniversalConstants().m_p)
    ]
    
    names = [u.name for u in universes]
    visualizer.plot_comparison(universes, names)
    
    # 5. ИТОГОВЫЙ ОТЧЕТ
    print("\n" + "="*60)
    print("📊 ИТОГОВЫЙ ОТЧЕТ ПО ОПТИМИЗАЦИИ")
    print("="*60)
    
    print(f"\n🌍 НАША ВСЕЛЕННАЯ:")
    print(f"   α = {1/137.036:.6f}")
    print(f"   m_p/m_p₀ = 1.000")
    
    our_analyzer = UniverseAnalyzer(UniverseParameters())
    _, our_score, our_metrics = our_analyzer.calculate_habitability_index()
    print(f"   Индекс пригодности = {our_score:.3f}")
    print(f"   Категория: HABITABLE")
    
    print(f"\n✨ ОПТИМАЛЬНАЯ ПО α:")
    print(f"   α = {opt_result['alpha']:.6f}")
    print(f"   Улучшение: {(opt_result['score']/our_score - 1)*100:.1f}%")
    print(f"   Индекс пригодности = {opt_result['score']:.3f}")
    print(f"   Категория: {opt_result['index']}")
    
    print(f"\n🌟 ГЛОБАЛЬНЫЙ ОПТИМУМ (α + m_p):")
    print(f"   α = {opt_2d['alpha']:.6f}")
    print(f"   m_p/m_p₀ = {opt_2d['m_p_ratio']:.3f}")
    print(f"   Улучшение: {(opt_2d['score']/our_score - 1)*100:.1f}%")
    print(f"   Индекс пригодности = {opt_2d['score']:.3f}")
    print(f"   Категория: {opt_2d['index']}")
    
    print(f"\n📈 КЛЮЧЕВЫЕ ВЫВОДЫ:")
    print(f"   1. Оптимальная α ≈ {opt_result['alpha']:.4f} (наше значение {1/137.036:.4f})")
    print(f"   2. Диапазон пригодности: α ∈ [{grid['alphas'][np.any(grid['score_map']>0.6, axis=1)].min():.4f}, "
          f"{grid['alphas'][np.any(grid['score_map']>0.6, axis=1)].max():.4f}]")
    print(f"   3. Оптимальная масса протона: m_p/m_p₀ ≈ {opt_2d['m_p_ratio']:.2f}")
    print(f"   4. Наша Вселенная находится в {'оптимальной' if our_score>0.8 else 'хорошей'} зоне")
    
    print("\n" + "="*60)
    print("🎉 ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
    print("="*60)
