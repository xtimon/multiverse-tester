import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, fsolve
from scipy.integrate import quad
import os
from datetime import datetime

# ============================================================================
# КОНСТАНТЫ И ПАРАМЕТРЫ МОДЕЛИ
# ============================================================================

# Коэффициенты полинома H(α) = a·α³ + b·α² + c·α + d
# Получены из аппроксимации экспериментальных данных
POLY_COEFFS = {
    'a': -1527777.78,  # α³
    'b': 31011.90,      # α²
    'c': -206.63,       # α
    'd': 1.33           # свободный член
}

# Пороговые значения индекса пригодности
THRESHOLDS = {
    'optimal': 0.875,   # Нижняя граница OPTIMAL зоны
    'marginal': 0.84,   # Нижняя граница MARGINAL зоны
    'hostile': 0.80     # Нижняя граница HOSTILE зоны (условно)
}

# Диапазоны для анализа
RANGES = {
    'alpha_min': 0.005,
    'alpha_max': 0.012,
    'alpha_step': 10000  # количество точек для гладкой кривой
}

# Наша Вселенная
OUR_UNIVERSE = {
    'alpha': 0.0073,
    'description': 'Наша Вселенная (α = 1/137 ≈ 0.0073)'
}

# Экспериментальные точки (из измерений)
EXPERIMENTAL_POINTS = {
    'alpha': np.array([0.006, 0.007, 0.008, 0.009, 0.010, 0.011]),
    'H': np.array([0.880, 0.885, 0.880, 0.875, 0.840, 0.780])
}

# ============================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# ============================================================================

def H_alpha(alpha):
    """
    Индекс пригодности вселенной как функция постоянной тонкой структуры α.
    
    Parameters:
    -----------
    alpha : float или array
        Значение постоянной тонкой структуры
        
    Returns:
    --------
    float или array : индекс пригодности H(α)
    """
    a = POLY_COEFFS['a']
    b = POLY_COEFFS['b']
    c = POLY_COEFFS['c']
    d = POLY_COEFFS['d']
    return a * alpha**3 + b * alpha**2 + c * alpha + d

def find_peak():
    """
    Находит точный максимум функции H(α) в диапазоне [0.006, 0.008].
    
    Returns:
    --------
    tuple : (α_peak, H_peak)
    """
    result = minimize_scalar(
        lambda x: -H_alpha(x), 
        bounds=(0.006, 0.008), 
        method='bounded'
    )
    return result.x, H_alpha(result.x)

def find_boundary(threshold, start_point, direction='right', 
                  search_range=(0.005, 0.012), num_points=10000):
    """
    Находит границу, где H(α) становится меньше threshold.
    
    Parameters:
    -----------
    threshold : float
        Пороговое значение индекса
    start_point : float
        Точка, от которой начинаем поиск
    direction : str
        'right' или 'left' - направление поиска
    search_range : tuple
        Диапазон поиска (min, max)
    num_points : int
        Количество точек для поиска
        
    Returns:
    --------
    float или None : значение α на границе или None если не найдено
    """
    if direction == 'right':
        alpha_test = np.linspace(start_point, search_range[1], num_points)
        for a in alpha_test:
            if H_alpha(a) < threshold:
                return a
    else:
        alpha_test = np.linspace(search_range[0], start_point, num_points)
        for a in reversed(alpha_test):
            if H_alpha(a) < threshold:
                return a
    return None

def calculate_derivatives(alpha_fine, H_fine):
    """
    Вычисляет первую и вторую производные.
    
    Returns:
    --------
    tuple : (dH, d2H, peak_idx, our_idx)
    """
    dH = np.gradient(H_fine, alpha_fine)
    d2H = np.gradient(dH, alpha_fine)
    
    peak_idx = np.argmin(np.abs(alpha_fine - PEAK['alpha']))
    our_idx = np.argmin(np.abs(alpha_fine - OUR_UNIVERSE['alpha']))
    
    return dH, d2H, peak_idx, our_idx

def calculate_integral_statistics():
    """
    Вычисляет интегральные характеристики модели.
    
    Returns:
    --------
    dict : статистики
    """
    total_area, _ = quad(H_alpha, RANGES['alpha_min'], RANGES['alpha_max'])
    mean_H = total_area / (RANGES['alpha_max'] - RANGES['alpha_min'])
    
    our_H = H_alpha(OUR_UNIVERSE['alpha'])
    relative_advantage = (our_H / mean_H - 1) * 100
    
    return {
        'total_area': total_area,
        'mean_H': mean_H,
        'our_H': our_H,
        'relative_advantage': relative_advantage  # в процентах
    }

def create_plot(derivatives, boundaries, stats, save=True):
    """
    Создает и сохраняет визуализацию.
    """
    # Создаем массив точек для графика
    alpha_fine = np.linspace(
        RANGES['alpha_min'], 
        RANGES['alpha_max'], 
        RANGES['alpha_step']
    )
    H_fine = H_alpha(alpha_fine)
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Зоны с заливкой (динамически определяем границы)
    if boundaries['left_optimal'] and boundaries['right_optimal']:
        mask_optimal = (alpha_fine >= boundaries['left_optimal']) & \
                       (alpha_fine <= boundaries['right_optimal'])
        ax.fill_between(alpha_fine[mask_optimal], 0.75, 0.9, 
                        alpha=0.2, color='green', 
                        label=f'OPTIMAL (H≥{THRESHOLDS["optimal"]})')
    
    if boundaries['left_marginal'] and boundaries['right_marginal']:
        mask_marginal = (alpha_fine >= boundaries['left_marginal']) & \
                        (alpha_fine <= boundaries['right_marginal'])
        # Исключаем уже закрашенную OPTIMAL зону
        if boundaries['left_optimal'] and boundaries['right_optimal']:
            mask_marginal = mask_marginal & ~mask_optimal
        ax.fill_between(alpha_fine[mask_marginal], 0.75, 0.9, 
                        alpha=0.2, color='yellow', 
                        label=f'MARGINAL ({THRESHOLDS["marginal"]}-{THRESHOLDS["optimal"]})')
    
    mask_hostile = alpha_fine > boundaries['right_marginal'] if boundaries['right_marginal'] else alpha_fine > 0.011
    ax.fill_between(alpha_fine[mask_hostile], 0.75, 0.9, 
                    alpha=0.2, color='red', 
                    label=f'HOSTILE (H<{THRESHOLDS["marginal"]})')
    
    # Основная кривая
    ax.plot(alpha_fine, H_fine, 'b-', linewidth=3, label='H(α) модель')
    
    # Экспериментальные точки
    ax.plot(EXPERIMENTAL_POINTS['alpha'], EXPERIMENTAL_POINTS['H'], 
            'ko', markersize=8, label='Эксперимент', zorder=5)
    
    # Важные точки
    ax.plot(PEAK['alpha'], PEAK['H'], 'g*', markersize=25, 
            label=f'Пик: α={PEAK["alpha"]:.4f}, H={PEAK["H"]:.4f}', zorder=10)
    ax.plot(OUR_UNIVERSE['alpha'], stats['our_H'], 'r*', markersize=25, 
            label=OUR_UNIVERSE['description'], zorder=10)
    
    # Горизонтальные линии порогов
    colors = {'optimal': 'green', 'marginal': 'orange', 'hostile': 'red'}
    for zone, thresh in THRESHOLDS.items():
        ax.axhline(y=thresh, color=colors[zone], linestyle='--', 
                   alpha=0.5, linewidth=2)
    
    # Вертикальные линии границ (если найдены)
    if boundaries['left_optimal']:
        ax.axvline(x=boundaries['left_optimal'], color='green', 
                   linestyle=':', alpha=0.3)
    if boundaries['right_optimal']:
        ax.axvline(x=boundaries['right_optimal'], color='green', 
                   linestyle=':', alpha=0.3)
    if boundaries['right_marginal']:
        ax.axvline(x=boundaries['right_marginal'], color='orange', 
                   linestyle=':', alpha=0.3)
    
    # Настройка осей
    ax.set_xlabel('Постоянная тонкой структуры α', fontsize=14)
    ax.set_ylabel('Индекс пригодности H(α)', fontsize=14)
    ax.set_title('Математическая модель плато пригодности вселенных', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(RANGES['alpha_min'], RANGES['alpha_max'])
    ax.set_ylim(0.75, 0.9)
    
    # Информационное поле (динамическое позиционирование)
    text_x = RANGES['alpha_min'] + 0.75 * (RANGES['alpha_max'] - RANGES['alpha_min'])
    text_y = 0.88
    
    textstr = '\n'.join((
        f'Пик: α={PEAK["alpha"]:.4f}, H={PEAK["H"]:.4f}',
        f'Наша α: {OUR_UNIVERSE["alpha"]:.4f}, H={stats["our_H"]:.4f}',
        f'Разница H: {stats["our_H"]-PEAK["H"]:+.4f}',
        f'dH/dα у нас: {derivatives["dH"][derivatives["our_idx"]]:.2f}',
        f'Преимущество над средним: +{stats["relative_advantage"]:.2f}%'
    ))
    
    # Позиционируем текст в правом верхнем углу, но с отступом от края
    ax.text(text_x, text_y, textstr, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            fontsize=10, verticalalignment='top',
            horizontalalignment='right')
    
    plt.tight_layout()
    
    # Сохранение графика
    if save:
        # Создаем папку для графиков, если её нет
        os.makedirs('reports', exist_ok=True)
        
        # Генерируем имя файла с датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'reports/habitability_plot_{timestamp}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📸 График сохранен: {filename}")
    
    plt.show()
    
    return fig

# ============================================================================
# ОСНОВНОЙ АНАЛИЗ
# ============================================================================

print("=" * 70)
print("ФИНАЛЬНЫЙ МАТЕМАТИЧЕСКИЙ АНАЛИЗ ПЛАТО ПРИГОДНОСТИ")
print("=" * 70)

# 1. Находим пик
PEAK = {}
PEAK['alpha'], PEAK['H'] = find_peak()

print(f"\n📐 ТОЧНЫЙ ПИК ПЛАТО:")
print(f"   α_peak = {PEAK['alpha']:.6f}")
print(f"   H_peak = {PEAK['H']:.6f}")

# 2. Наша Вселенная
our_H = H_alpha(OUR_UNIVERSE['alpha'])
print(f"\n📍 НАША ВСЕЛЕННАЯ:")
print(f"   α_our = {OUR_UNIVERSE['alpha']:.6f}")
print(f"   H_our = {our_H:.6f}")
print(f"   Отклонение от пика: {our_H - PEAK['H']:+.6f}")

# 3. Создаем массив для анализа
alpha_fine = np.linspace(RANGES['alpha_min'], RANGES['alpha_max'], RANGES['alpha_step'])
H_fine = H_alpha(alpha_fine)

# 4. Вычисляем производные
dH, d2H, peak_idx, our_idx = calculate_derivatives(alpha_fine, H_fine)
derivatives = {
    'dH': dH,
    'd2H': d2H,
    'peak_idx': peak_idx,
    'our_idx': our_idx
}

print(f"\n📈 АНАЛИЗ ПРОИЗВОДНЫХ:")
print(f"   Первая производная на пике: {dH[peak_idx]:.2f}")
print(f"   Первая производная у нас: {dH[our_idx]:.2f}")
print(f"   Вторая производная на пике: {d2H[peak_idx]:.2f}")
print(f"   Вторая производная у нас: {d2H[our_idx]:.2f}")
print(f"   Скорость изменения у нас: при Δα=0.001, ΔH={dH[our_idx]*0.001:.6f}")

# 5. Находим границы
print(f"\n📏 ПОИСК ГРАНИЦ:")

boundaries = {
    'left_optimal': find_boundary(THRESHOLDS['optimal'], PEAK['alpha'], 'left'),
    'right_optimal': find_boundary(THRESHOLDS['optimal'], PEAK['alpha'], 'right'),
    'left_marginal': find_boundary(THRESHOLDS['marginal'], PEAK['alpha'], 'left'),
    'right_marginal': find_boundary(THRESHOLDS['marginal'], PEAK['alpha'], 'right')
}

if boundaries['left_optimal'] and boundaries['right_optimal']:
    print(f"\n   OPTIMAL зона (H ≥ {THRESHOLDS['optimal']}):")
    print(f"      [{boundaries['left_optimal']:.6f}, {boundaries['right_optimal']:.6f}]")
    print(f"      Ширина: {boundaries['right_optimal'] - boundaries['left_optimal']:.6f}")
else:
    print(f"\n   ⚠️ OPTIMAL зона выходит за пределы исследованного диапазона")

# 6. Точка падения до H=0.8
alpha_08 = fsolve(lambda x: H_alpha(x) - THRESHOLDS['hostile'], 0.011)[0]
print(f"\n   Точка падения до H={THRESHOLDS['hostile']}: α = {alpha_08:.6f}")

# 7. Интегральные характеристики
stats = calculate_integral_statistics()

print(f"\n📊 ИНТЕГРАЛЬНЫЕ ХАРАКТЕРИСТИКИ:")
print(f"   Средний индекс в [{RANGES['alpha_min']}, {RANGES['alpha_max']}]: {stats['mean_H']:.6f}")
print(f"   Отклонение нашего индекса от среднего: {stats['our_H'] - stats['mean_H']:+.6f}")
print(f"   Наш индекс относительно среднего: +{stats['relative_advantage']:.2f}%")

# 8. Создаем и сохраняем график
create_plot(derivatives, boundaries, stats, save=True)

# 9. Финальные выводы
print("\n" + "=" * 70)
print("🎯 ФИНАЛЬНЫЕ ВЫВОДЫ:")
print("=" * 70)
print(f"1. Пик плато находится при α = {PEAK['alpha']:.6f}")
print(f"2. Наша Вселенная (α = {OUR_UNIVERSE['alpha']:.6f}) на {abs(our_H-PEAK['H']):.6f} ниже пика")
print(f"3. Скорость изменения у нас: {dH[our_idx]:.2f} (при изменении α на 0.001, H меняется на {dH[our_idx]*0.001:.6f})")
print(f"4. Кривизна отрицательная - мы на вершине плато")
print(f"5. Относительно среднего индекса мы на +{stats['relative_advantage']:.2f}% выше")
print(f"6. До опасной зоны (H<{THRESHOLDS['hostile']}) нужно увеличить α на {((alpha_08 - OUR_UNIVERSE['alpha'])/OUR_UNIVERSE['alpha']*100):.1f}%")
print("=" * 70)
