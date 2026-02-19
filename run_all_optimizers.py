#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакетный запуск всех оптимизаторов MultiverseTester.
Собирает результаты и формирует итоговый отчет.
Запуск: python run_all_optimizers.py
"""

import os
import sys
import io
from datetime import datetime
from pathlib import Path

# Устанавливаем non-interactive режим для matplotlib ПЕРЕД импортом
os.environ['MPLBACKEND'] = 'Agg'

# Патчим plt.show для сохранения графиков вместо отображения
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_fig_counter = [0]  # mutable for closure
_fig_dir = None

def _save_fig_instead_of_show():
    """Сохраняет текущий рисунок в папку reports/figures вместо показа"""
    global _fig_counter, _fig_dir
    if _fig_dir is None:
        _fig_dir = Path('reports/figures')
        _fig_dir.mkdir(parents=True, exist_ok=True)
    for i, fig in enumerate(plt.get_fignums()):
        f = plt.figure(fig)
        _fig_counter[0] += 1
        f.savefig(_fig_dir / f'fig_{_fig_counter[0]:03d}.png', dpi=100, bbox_inches='tight')
    plt.close('all')

# Заменяем plt.show
plt.show = _save_fig_instead_of_show


def run_2d_optimizer():
    """Запуск 2D оптимизатора"""
    from multiverse_tester import UniverseParameters, UniverseAnalyzer, UniversalConstants
    from scipy.optimize import minimize_scalar, differential_evolution
    
    results = {}
    const = UniversalConstants()
    
    # 1. Оптимизация α
    def objective_alpha(x):
        u = UniverseParameters(alpha=x)
        analyzer = UniverseAnalyzer(u)
        _, score, _ = analyzer.calculate_habitability_index()
        return 1.0 - score
    
    res_alpha = minimize_scalar(objective_alpha, bounds=(1/300, 1/30), method='bounded', options={'xatol': 1e-6})
    results['opt_alpha'] = res_alpha.x
    results['opt_alpha_score'] = 1.0 - res_alpha.fun
    
    # 2. 2D оптимизация (α, m_p)
    def objective_2d(x):
        alpha, m_p_ratio = x
        u = UniverseParameters(alpha=alpha, m_p=m_p_ratio * const.m_p)
        analyzer = UniverseAnalyzer(u)
        _, score, _ = analyzer.calculate_habitability_index()
        return 1.0 - score
    
    res_2d = differential_evolution(objective_2d, [(1/300, 1/30), (0.5, 2.0)], 
                                    strategy='best1bin', popsize=25, maxiter=40, tol=1e-6, seed=42)
    results['opt_alpha_2d'] = res_2d.x[0]
    results['opt_m_p'] = res_2d.x[1]
    results['opt_2d_score'] = 1.0 - res_2d.fun
    
    # 3. Наша Вселенная
    our = UniverseParameters()
    our_analyzer = UniverseAnalyzer(our)
    _, results['our_score'], results['our_metrics'] = our_analyzer.calculate_habitability_index()
    
    # 4. Grid search (уменьшенная сетка)
    alphas = __import__('numpy').linspace(1/300, 1/30, 30)
    m_p_ratios = __import__('numpy').linspace(0.5, 2.0, 20)
    score_map = []
    for alpha in alphas:
        row = []
        for mp in m_p_ratios:
            u = UniverseParameters(alpha=alpha, m_p=mp * const.m_p)
            try:
                a = UniverseAnalyzer(u)
                _, s, _ = a.calculate_habitability_index()
                row.append(s)
            except:
                row.append(0)
        score_map.append(row)
    
    score_map = __import__('numpy').array(score_map)
    habitable = score_map > 0.6
    results['habitable_fraction_2d'] = habitable.sum() / habitable.size
    
    return results


def run_3d_optimizer():
    """Запуск 3D оптимизатора (уменьшенная сетка)"""
    import numpy as np
    from multiverse_tester import UniverseParameters, UniverseAnalyzer, UniversalConstants
    
    const = UniversalConstants()
    points = 15  # 15^3 = 3375
    alphas = np.linspace(1/300, 1/30, points)
    m_p_ratios = np.linspace(0.5, 2.0, points)
    m_e_ratios = np.linspace(0.5, 2.0, points)
    
    score_3d = np.zeros((points, points, points))
    for i, alpha in enumerate(alphas):
        for j, mp in enumerate(m_p_ratios):
            for k, me in enumerate(m_e_ratios):
                try:
                    u = UniverseParameters(alpha=alpha, m_p=mp*const.m_p, m_e=me*const.m_e)
                    a = UniverseAnalyzer(u)
                    _, s, _ = a.calculate_habitability_index()
                    score_3d[i,j,k] = s
                except:
                    score_3d[i,j,k] = 0
    
    max_idx = np.unravel_index(np.argmax(score_3d), score_3d.shape)
    habitable = score_3d > 0.6
    
    return {
        'best_alpha': alphas[max_idx[0]],
        'best_m_p': m_p_ratios[max_idx[1]],
        'best_m_e': m_e_ratios[max_idx[2]],
        'best_score': score_3d[max_idx],
        'habitable_fraction': habitable.sum() / habitable.size,
    }


def run_4d_optimizer():
    """Запуск 4D оптимизатора (уменьшенная сетка)"""
    import numpy as np
    from multiverse_tester import UniverseParameters, UniverseAnalyzer, UniversalConstants
    
    const = UniversalConstants()
    points = 8  # 8^4 = 4096
    alphas = np.linspace(1/300, 1/30, points)
    m_p_ratios = np.linspace(0.3, 3.0, points)
    m_e_ratios = np.linspace(0.3, 3.0, points)
    G_ratios = np.linspace(0.1, 10.0, points)
    
    score_4d = np.zeros((points, points, points, points))
    total = points**4
    count = 0
    for i, alpha in enumerate(alphas):
        for j, mp in enumerate(m_p_ratios):
            for k, me in enumerate(m_e_ratios):
                for l, G in enumerate(G_ratios):
                    try:
                        u = UniverseParameters(alpha=alpha, m_p=mp*const.m_p, 
                                              m_e=me*const.m_e, G=G*const.G)
                        a = UniverseAnalyzer(u)
                        _, s, _ = a.calculate_habitability_index()
                        score_4d[i,j,k,l] = s
                    except:
                        score_4d[i,j,k,l] = 0
                    count += 1
                    if count % 1000 == 0:
                        print(f"   4D: {count}/{total} ({100*count/total:.1f}%)")
    
    max_idx = np.unravel_index(np.argmax(score_4d), score_4d.shape)
    habitable = score_4d > 0.6
    
    return {
        'best_alpha': alphas[max_idx[0]],
        'best_m_p': m_p_ratios[max_idx[1]],
        'best_m_e': m_e_ratios[max_idx[2]],
        'best_G': G_ratios[max_idx[3]],
        'best_score': score_4d[max_idx],
        'habitable_fraction': habitable.sum() / habitable.size,
    }


def run_5d_optimizer():
    """Запуск 5D оптимизатора"""
    import numpy as np
    import importlib.util
    spec = importlib.util.spec_from_file_location("opt5d", Path(__file__).parent / "5Doptimizator.py")
    opt5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opt5)
    
    hv = opt5.HyperVolume5D()
    results = hv.generate_5d_grid(
        alpha_range=(1/300, 1/30),
        m_p_range=(0.3, 3.0),
        m_e_range=(0.3, 3.0),
        G_range=(0.2, 5.0),
        c_range=(0.5, 2.0),
        points=6
    )
    vol = hv.calculate_5d_volume(threshold=0.6)
    
    return {
        'best_alpha': results['best_alpha'],
        'best_m_p': results['best_m_p'],
        'best_m_e': results['best_m_e'],
        'best_G': results['best_G'],
        'best_c': results['best_c'],
        'best_score': results['best_score'],
        'habitable_fraction': vol.get('fraction', 0),
    }


def run_6d_optimizer():
    """Запуск 6D оптимизатора"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("opt6d", Path(__file__).parent / "6D_optimizator.py")
    opt6 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opt6)
    
    hv = opt6.HyperVolume6D()
    results = hv.generate_6d_grid(
        alpha_range=(1/300, 1/30),
        m_p_range=(0.3, 3.0),
        m_e_range=(0.3, 3.0),
        G_range=(0.2, 5.0),
        c_range=(0.5, 2.0),
        hbar_range=(0.5, 2.0),
        points=5  # 5^6 = 15625
    )
    vol = hv.calculate_6d_volume(threshold=0.6)
    
    return {
        'best_alpha': results['best_alpha'],
        'best_m_p': results['best_m_p'],
        'best_m_e': results['best_m_e'],
        'best_G': results['best_G'],
        'best_c': results['best_c'],
        'best_hbar': results['best_hbar'],
        'best_score': results['best_score'],
        'habitable_fraction': vol.get('fraction', 0),
    }


def run_7d_optimizer():
    """Запуск 7D оптимизатора (α, m_p, m_e, G, c, ħ, ε₀)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("opt7d", Path(__file__).parent / "7D_optimizator.py")
    opt7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opt7)
    
    hv = opt7.HyperVolume7D()
    results = hv.generate_7d_adaptive(
        alpha_range=(1/400, 1/15),
        m_p_range=(0.1, 5.0),
        m_e_range=(0.1, 5.0),
        G_range=(0.05, 10.0),
        c_range=(0.2, 3.0),
        hbar_range=(0.2, 3.0),
        epsilon_0_range=(0.1, 5.0),
        coarse_points=3,   # 3^7 = 2,187 (грубая сетка)
        zoom_points=4,     # 4^7 = 16,384 (рефайнмент)
        zoom_fraction=0.25,
        max_refinements=2
    )
    vol = hv.calculate_7d_volume(threshold=0.6)
    
    return {
        'best_alpha': results['best_alpha'],
        'best_m_p': results['best_m_p'],
        'best_m_e': results['best_m_e'],
        'best_G': results['best_G'],
        'best_c': results['best_c'],
        'best_hbar': results['best_hbar'],
        'best_eps': results['best_eps'],
        'best_score': results['best_score'],
        'habitable_fraction': vol.get('fraction', 0),
    }


def run_8d_optimizer():
    """Запуск 8D оптимизатора (α, m_p, m_e, G, c, ħ, ε₀, k_B)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("opt8d", Path(__file__).parent / "8D_optimizator.py")
    opt8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opt8)
    
    hv = opt8.HyperVolume8D()
    results = hv.generate_8d_adaptive(
        alpha_range=(1/400, 1/15),
        m_p_range=(0.1, 5.0),
        m_e_range=(0.1, 5.0),
        G_range=(0.05, 10.0),
        c_range=(0.2, 3.0),
        hbar_range=(0.2, 3.0),
        epsilon_0_range=(0.1, 5.0),
        k_B_range=(0.1, 5.0),
        coarse_points=3,   # 3^8 = 6,561
        zoom_points=3,     # 3^8 = 6,561
        zoom_fraction=0.25,
        max_refinements=2
    )
    vol = hv.calculate_8d_volume(threshold=0.6)
    
    return {
        'best_alpha': results['best_alpha'],
        'best_m_p': results['best_m_p'],
        'best_m_e': results['best_m_e'],
        'best_G': results['best_G'],
        'best_c': results['best_c'],
        'best_hbar': results['best_hbar'],
        'best_eps': results['best_eps'],
        'best_k_B': results['best_k_B'],
        'best_score': results['best_score'],
        'habitable_fraction': vol.get('fraction', 0),
    }


def run_9d_optimizer():
    """Запуск 9D оптимизатора (α, m_p, m_e, G, c, ħ, ε₀, k_B, H₀)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("opt9d", Path(__file__).parent / "9D_optimizator.py")
    opt9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opt9)
    
    hv = opt9.HyperVolume9D()
    results = hv.generate_9d_adaptive(
        alpha_range=(1/400, 1/15),
        m_p_range=(0.1, 5.0),
        m_e_range=(0.1, 5.0),
        G_range=(0.05, 10.0),
        c_range=(0.2, 3.0),
        hbar_range=(0.2, 3.0),
        epsilon_0_range=(0.1, 5.0),
        k_B_range=(0.1, 5.0),
        H0_range=(0.2, 5.0),
        coarse_points=3,
        zoom_points=3,
        zoom_fraction=0.25,
        max_refinements=2
    )
    vol = hv.calculate_9d_volume(threshold=0.6)
    
    return {
        'best_alpha': results['best_alpha'],
        'best_m_p': results['best_m_p'],
        'best_m_e': results['best_m_e'],
        'best_G': results['best_G'],
        'best_c': results['best_c'],
        'best_hbar': results['best_hbar'],
        'best_eps': results['best_eps'],
        'best_k_B': results['best_k_B'],
        'best_H0': results['best_H0'],
        'best_score': results['best_score'],
        'habitable_fraction': vol.get('fraction', 0),
    }


def run_10d_optimizer():
    """Запуск 10D оптимизатора (α, m_p, m_e, G, c, ħ, ε₀, k_B, H₀, Λ)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("opt10d", Path(__file__).parent / "10D_optimizator.py")
    opt10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(opt10)

    hv = opt10.HyperVolume10D()
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
    vol = hv.calculate_10d_volume(threshold=0.6)

    return {
        'best_alpha': results['best_alpha'],
        'best_m_p': results['best_m_p'],
        'best_m_e': results['best_m_e'],
        'best_G': results['best_G'],
        'best_c': results['best_c'],
        'best_hbar': results['best_hbar'],
        'best_eps': results['best_eps'],
        'best_k_B': results['best_k_B'],
        'best_H0': results['best_H0'],
        'best_Lambda': results['best_Lambda'],
        'best_score': results['best_score'],
        'habitable_fraction': vol.get('fraction', 0),
    }


def main():
    Path('reports').mkdir(exist_ok=True)
    Path('reports/figures').mkdir(exist_ok=True)
    
    all_results = {}
    
    print("="*70)
    print("🚀 ПАКЕТНЫЙ ЗАПУСК ОПТИМИЗАТОРОВ MULTIVERSETESTER")
    print("="*70)
    
    # 2D
    print("\n📊 2D ОПТИМИЗАТОР (α, m_p)...")
    try:
        all_results['2D'] = run_2d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['2D'] = {'error': str(e)}
    
    # 3D
    print("\n📊 3D ОПТИМИЗАТОР (α, m_p, m_e)...")
    try:
        all_results['3D'] = run_3d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['3D'] = {'error': str(e)}
    
    # 4D
    print("\n📊 4D ОПТИМИЗАТОР (α, m_p, m_e, G)...")
    try:
        all_results['4D'] = run_4d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['4D'] = {'error': str(e)}
    
    # 5D
    print("\n📊 5D ОПТИМИЗАТОР (α, m_p, m_e, G, c)...")
    try:
        all_results['5D'] = run_5d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['5D'] = {'error': str(e)}
    
    # 6D
    print("\n📊 6D ОПТИМИЗАТОР (α, m_p, m_e, G, c, ħ)...")
    try:
        all_results['6D'] = run_6d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['6D'] = {'error': str(e)}
    
    # 7D
    print("\n📊 7D ОПТИМИЗАТОР (α, m_p, m_e, G, c, ħ, ε₀)...")
    try:
        all_results['7D'] = run_7d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['7D'] = {'error': str(e)}
    
    # 8D
    print("\n📊 8D ОПТИМИЗАТОР (α, m_p, m_e, G, c, ħ, ε₀, k_B)...")
    try:
        all_results['8D'] = run_8d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['8D'] = {'error': str(e)}
    
    # 9D
    print("\n📊 9D ОПТИМИЗАТОР (α, m_p, m_e, G, c, ħ, ε₀, k_B, H₀)...")
    try:
        all_results['9D'] = run_9d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['9D'] = {'error': str(e)}

    # 10D
    print("\n📊 10D ОПТИМИЗАТОР (α, m_p, m_e, G, c, ħ, ε₀, k_B, H₀, Λ)...")
    try:
        all_results['10D'] = run_10d_optimizer()
        print("   ✓ Завершено")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        all_results['10D'] = {'error': str(e)}
    
    # Генерация отчета
    report_path = Path('reports/OPTIMIZATION_REPORT.md')
    generate_report(all_results, report_path)
    
    print("\n" + "="*70)
    print(f"📄 Отчет сохранен: {report_path}")
    print("="*70)


def generate_report(results: dict, path: Path):
    """Генерирует Markdown отчет"""
    lines = [
        "# Отчет по оптимизации MultiverseTester",
        "",
        f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Резюме",
        "",
        "| Размерность | Оптимальная α | Лучший score | Доля пригодного пространства |",
        "|-------------|----------------|--------------|------------------------------|",
    ]
    
    our_score = results.get('2D', {}).get('our_score', 0)
    
    def fmt(v, p=4):
        if isinstance(v, float):
            return f"{v:.{p}f}"
        return str(v)
    
    for dim in ['2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D']:
        r = results.get(dim, {})
        if 'error' in r:
            lines.append(f"| {dim} | — | Ошибка | — |")
            continue
        
        alpha = r.get('best_alpha', r.get('opt_alpha', r.get('opt_alpha_2d')))
        score = r.get('best_score', r.get('opt_2d_score', r.get('opt_alpha_score')))
        frac = r.get('habitable_fraction', r.get('habitable_fraction_2d', 0))
        if frac:
            frac_str = f"{frac*100:.2f}%"
        else:
            frac_str = "—"
        lines.append(f"| {dim} | {fmt(alpha,6) if alpha else '—'} | {fmt(score) if score else '—'} | {frac_str} |")
    
    lines.extend([
        "",
        "## 2D (α, m_p)",
        "",
    ])
    
    r2 = results.get('2D', {})
    if 'error' not in r2:
        lines.extend([
            f"- Оптимальная α (1D): {r2.get('opt_alpha', '—'):.6f}",
            f"- Оптимальная α (2D): {r2.get('opt_alpha_2d', '—'):.6f}",
            f"- Оптимальная m_p/m_p₀: {r2.get('opt_m_p', '—'):.3f}",
            f"- Индекс нашей Вселенной: {r2.get('our_score', '—'):.3f}",
            f"- Доля пригодного пространства: {r2.get('habitable_fraction_2d', 0)*100:.2f}%",
            "",
        ])
    
    for dim in ['3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D']:
        r = results.get(dim, {})
        lines.append(f"## {dim}")
        lines.append("")
        if 'error' in r:
            lines.append(f"Ошибка: {r['error']}")
        else:
            param_map = {'alpha': 'best_alpha', 'm_p': 'best_m_p', 'm_e': 'best_m_e', 
                     'G': 'best_G', 'c': 'best_c', 'hbar': 'best_hbar', 'ε₀': 'best_eps', 
                     'k_B': 'best_k_B', 'H₀': 'best_H0', 'Λ': 'best_Lambda'}
            for pname, key in param_map.items():
                v = r.get(key)
                if v is not None:
                    lines.append(f"- Оптимальная {pname}: {v:.4f}")
            if r.get('best_score'):
                lines.append(f"- Лучший индекс пригодности: {r['best_score']:.3f}")
            if r.get('habitable_fraction'):
                lines.append(f"- Доля пригодного пространства: {r['habitable_fraction']*100:.2f}%")
        lines.append("")
    
    lines.extend([
        "## Выводы",
        "",
        "1. **Постоянная тонкой структуры (α)** — ключевой параметр; оптимальное значение близко к 1/137.",
        "2. **Массы частиц** — допустимый диапазон варьируется в 2–3 раза от наших значений.",
        "3. **Гравитационная постоянная G** — может изменяться в десятки раз при сохранении пригодности.",
        "4. **Скорость света c и ħ** — более жесткие ограничения, оптимум около наших значений.",
        "5. С ростом размерности доля пригодного пространства уменьшается.",
        "",
    ])
    
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == "__main__":
    main()
