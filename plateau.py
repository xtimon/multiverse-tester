"""
ПОЛНЫЙ АНАЛИЗ ПЛАТО ПРИГОДНОСТИ ВСЕЛЕННЫХ
===========================================
Этапы:
1. Импорт библиотеки multiverse-tester
2. Генерация данных для разных значений α
3. Построение математической модели
4. Визуализация и анализ
5. Сохранение результатов
"""

import json
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import fsolve, minimize_scalar

# ============================================================================
# 1. ИМПОРТ БИБЛИОТЕКИ MULTIVERSE-TESTER
# ============================================================================

try:
    from multiverse_tester import UniverseAnalyzer, UniverseParameters

    print("✅ Библиотека multiverse-tester успешно импортирована")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("   Установите библиотеку: pip install multiverse-tester")
    raise SystemExit(1) from e

# ============================================================================
# 2. КОНСТАНТЫ И ПАРАМЕТРЫ
# ============================================================================

# Численные параметры
GRID_POINTS_FINE = 10_000
GRID_POINTS_PLOT = 1_000
PROGRESS_BAR_LENGTH = 30
PEAK_SEARCH_BOUNDS = (0.006, 0.008)

# Базовая конфигурация нашей Вселенной
OUR_UNIVERSE: dict[str, Any] = {
    "name": "Наша Вселенная",
    "alpha": 1 / 137.036,  # 0.007297
    "m_p": 1.6726219e-27,
    "m_e": 9.1093837e-31,
    "G": 6.6743e-11,
    "c": 299792458,
    "hbar": 1.0545718e-34,
    "epsilon_0": 8.8541878128e-12,
    "k_B": 1.380649e-23,
    "H_0": 67.4,
    "Lambda": 1e-52,
}

# Параметры для исследования
STUDY_PARAMS: dict[str, Any] = {
    "alpha_range": (0.005, 0.012),
    "alpha_points": 20,
    "fixed_params": {
        "m_p": OUR_UNIVERSE["m_p"],
        "m_e": OUR_UNIVERSE["m_e"],
        "G": OUR_UNIVERSE["G"],
        "c": OUR_UNIVERSE["c"],
        "hbar": OUR_UNIVERSE["hbar"],
        "epsilon_0": OUR_UNIVERSE["epsilon_0"],
        "k_B": OUR_UNIVERSE["k_B"],
        "H_0": OUR_UNIVERSE["H_0"],
        "Lambda": OUR_UNIVERSE["Lambda"],
    },
}

THRESHOLDS = {
    "optimal": 0.875,
    "marginal": 0.84,
    "hostile": 0.80,
}

ZONE_COLORS = {"optimal": "green", "marginal": "orange", "hostile": "red"}
OUTPUT_DIR = "reports"

# ============================================================================
# 3. ГЕНЕРАЦИЯ ДАННЫХ С ПОМОЩЬЮ БИБЛИОТЕКИ
# ============================================================================


def _analyze_single_universe(alpha: float) -> tuple[float, str, Any]:
    """Анализирует одну вселенную с заданным α."""
    universe = UniverseParameters(
        name=f"α={alpha:.4f}",
        alpha=alpha,
        **STUDY_PARAMS["fixed_params"],
    )
    analyzer = UniverseAnalyzer(universe)
    index, score, metrics = analyzer.calculate_habitability_index()
    return score, index.name, metrics


def _print_progress(i: int, total: int, alpha: float, score: float) -> None:
    """Выводит прогресс-бар в консоль."""
    progress = (i + 1) / total * 100
    filled = int(PROGRESS_BAR_LENGTH * (i + 1) // total)
    bar = "█" * filled + "░" * (PROGRESS_BAR_LENGTH - filled)
    print(f"\r   [{bar}] {progress:.1f}% | α={alpha:.4f}, H={score:.4f}", end="")


def generate_alpha_data() -> dict[str, Any]:
    """
    Генерирует данные, варьируя α при фиксированных остальных параметрах.

    Returns:
    --------
    dict : {'alpha': array, 'H': array, 'detailed': list of dicts}
    """
    print("\n" + "=" * 70)
    print("🔬 ГЕНЕРАЦИЯ ДАННЫХ С ПОМОЩЬЮ MULTIVERSE-TESTER")
    print("=" * 70)

    alpha_min, alpha_max = STUDY_PARAMS["alpha_range"]
    n_points = STUDY_PARAMS["alpha_points"]
    alpha_values = np.linspace(alpha_min, alpha_max, n_points)

    results: dict[str, Any] = {"alpha": [], "H": [], "detailed": []}

    print(f"\n📊 Исследуем {n_points} значений α от {alpha_values[0]:.4f} до {alpha_values[-1]:.4f}")
    print("-" * 60)

    for i, alpha in enumerate(alpha_values):
        score, category, metrics = _analyze_single_universe(alpha)
        results["alpha"].append(alpha)
        results["H"].append(score)
        results["detailed"].append({
            "alpha": alpha,
            "score": score,
            "category": category,
            "metrics": metrics,
        })
        _print_progress(i, n_points, alpha, score)

    print("\n" + "-" * 60)
    print(f"✅ Сгенерировано {len(results['alpha'])} точек данных")

    results["alpha"] = np.array(results["alpha"])
    results["H"] = np.array(results["H"])
    return results

# ============================================================================
# 4. ПОСТРОЕНИЕ МАТЕМАТИЧЕСКОЙ МОДЕЛИ
# ============================================================================


def _compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Вычисляет коэффициент детерминации R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def _format_polynomial_expression(coeffs: np.ndarray) -> str:
    """Форматирует коэффициенты полинома в читаемое выражение (для степени 3)."""
    if len(coeffs) == 4:  # cubic
        return f"H(α) = {coeffs[0]:.2f}·α³ + {coeffs[1]:.2f}·α² + {coeffs[2]:.2f}·α + {coeffs[3]:.2f}"
    terms = []
    for i, c in enumerate(coeffs):
        power = len(coeffs) - 1 - i
        symbol = "α" if power == 1 else f"α^{power}" if power > 1 else ""
        terms.append(f"{c:.2f}·{symbol}" if symbol else f"{c:.2f}")
    return "H(α) = " + " + ".join(terms)


def fit_polynomial(alpha_data: np.ndarray, H_data: np.ndarray, degree: int = 3) -> dict[str, Any]:
    """
    Аппроксимирует данные полиномом заданной степени.

    Returns:
    --------
    dict : информация о модели (coeffs, poly, degree, r_squared, expression)
    """
    print("\n" + "=" * 70)
    print("📐 ПОСТРОЕНИЕ МАТЕМАТИЧЕСКОЙ МОДЕЛИ")
    print("=" * 70)

    coeffs = np.polyfit(alpha_data, H_data, degree)
    poly = np.poly1d(coeffs)
    H_pred = poly(alpha_data)
    r_squared = _compute_r_squared(H_data, H_pred)

    model = {
        "coeffs": coeffs,
        "poly": poly,
        "degree": degree,
        "r_squared": r_squared,
        "expression": _format_polynomial_expression(coeffs),
    }

    print(f"\n📈 Модель (степень {degree}):")
    print(f"   {model['expression']}")
    print(f"   R² = {r_squared:.6f} (качество аппроксимации)")

    return model


def _find_boundary(
    poly: np.poly1d,
    threshold: float,
    start: float,
    direction: str,
) -> float | None:
    """Находит границу зоны по порогу пригодности."""
    alpha_min, alpha_max = STUDY_PARAMS["alpha_range"]
    test = np.linspace(alpha_min, alpha_max, GRID_POINTS_FINE)

    if direction == "right":
        mask = test >= start
        test = test[mask]
        for a in test:
            if poly(a) < threshold:
                return float(a)
    else:
        mask = test <= start
        test = test[mask]
        for a in reversed(test):
            if poly(a) < threshold:
                return float(a)
    return None


def _compute_derivatives(poly: np.poly1d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Вычисляет α, dH/dα и d²H/dα² на плотной сетке."""
    alpha_min, alpha_max = STUDY_PARAMS["alpha_range"]
    alpha_fine = np.linspace(alpha_min, alpha_max, GRID_POINTS_FINE)
    H_fine = poly(alpha_fine)
    dH = np.gradient(H_fine, alpha_fine)
    d2H = np.gradient(dH, alpha_fine)
    return alpha_fine, dH, d2H


def analyze_model(model: dict[str, Any], alpha_data: np.ndarray, H_data: np.ndarray) -> dict[str, Any]:
    """Проводит полный математический анализ модели."""
    poly = model["poly"]
    alpha_min, alpha_max = STUDY_PARAMS["alpha_range"]
    our_alpha = OUR_UNIVERSE["alpha"]

    # Пик модели
    result = minimize_scalar(
        lambda x: -poly(x),
        bounds=PEAK_SEARCH_BOUNDS,
        method="bounded",
    )
    peak_alpha, peak_H = result.x, poly(result.x)
    our_H = poly(our_alpha)

    # Производные
    alpha_fine, dH, d2H = _compute_derivatives(poly)
    peak_idx = np.argmin(np.abs(alpha_fine - peak_alpha))
    our_idx = np.argmin(np.abs(alpha_fine - our_alpha))

    # Границы зон
    boundaries = {
        "left_optimal": _find_boundary(poly, THRESHOLDS["optimal"], peak_alpha, "left"),
        "right_optimal": _find_boundary(poly, THRESHOLDS["optimal"], peak_alpha, "right"),
        "right_marginal": _find_boundary(poly, THRESHOLDS["marginal"], peak_alpha, "right"),
    }

    # Интегральные характеристики
    total_area, _ = quad(poly, alpha_min, alpha_max)
    mean_H = total_area / (alpha_max - alpha_min)
    relative_advantage = (our_H / mean_H - 1) * 100

    # Граница hostile
    alpha_hostile = float(fsolve(lambda x: poly(x) - THRESHOLDS["hostile"], 0.011)[0])

    return {
        "peak": {"alpha": peak_alpha, "H": peak_H},
        "our": {
            "alpha": our_alpha,
            "H": our_H,
            "deviation": our_H - peak_H,
        },
        "derivatives": {
            "dH_peak": dH[peak_idx],
            "dH_our": dH[our_idx],
            "d2H_peak": d2H[peak_idx],
            "d2H_our": d2H[our_idx],
            "slope_per_0_001": dH[our_idx] * 0.001,
        },
        "boundaries": boundaries,
        "integral": {
            "mean_H": mean_H,
            "relative_advantage": relative_advantage,
            "total_area": total_area,
        },
        "hostile_boundary": alpha_hostile,
    }

# ============================================================================
# 5. ВИЗУАЛИЗАЦИЯ И ОТЧЕТЫ
# ============================================================================


def _get_timestamp() -> str:
    """Возвращает текущую метку времени для имен файлов."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_analysis_report(model: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Форматирует текстовый отчет по результатам анализа."""
    b = analysis["boundaries"]
    left_opt = f"{b['left_optimal']:.4f}" if b["left_optimal"] is not None else "N/A"
    right_opt = f"{b['right_optimal']:.4f}" if b["right_optimal"] is not None else "N/A"
    right_marg = f"{b['right_marginal']:.4f}" if b["right_marginal"] is not None else "N/A"
    return f"""
    📊 РЕЗУЛЬТАТЫ АНАЛИЗА
    =====================

    📐 МОДЕЛЬ:
    {model['expression']}
    R² = {model['r_squared']:.6f}

    🎯 ПИК:
    α_peak = {analysis['peak']['alpha']:.6f}
    H_peak = {analysis['peak']['H']:.6f}

    🌍 НАША ВСЕЛЕННАЯ:
    α = {analysis['our']['alpha']:.6f}
    H = {analysis['our']['H']:.6f}
    ΔH = {analysis['our']['deviation']:+.6f}

    📈 ПРОИЗВОДНЫЕ:
    dH/dα у нас = {analysis['derivatives']['dH_our']:.2f}
    При Δα=0.001 → ΔH={analysis['derivatives']['slope_per_0_001']:.6f}

    📏 ГРАНИЦЫ:
    OPTIMAL: [{left_opt}, {right_opt}]
    MARGINAL до: {right_marg}
    HOSTILE при α > {analysis['hostile_boundary']:.4f}

    📊 ИНТЕГРАЛ:
    Средний H = {analysis['integral']['mean_H']:.4f}
    Мы выше среднего на {analysis['integral']['relative_advantage']:.2f}%
    """


def _plot_main_chart(ax: plt.Axes, data: dict, model: dict, analysis: dict) -> None:
    """Рисует основной график с данными, моделью и зонами."""
    alpha_min, alpha_max = STUDY_PARAMS["alpha_range"]
    alpha_fine = np.linspace(alpha_min, alpha_max, GRID_POINTS_PLOT)
    H_fine = model["poly"](alpha_fine)
    b = analysis["boundaries"]
    right_opt = b["right_optimal"] or alpha_max
    right_marg = b["right_marginal"] or alpha_max

    if b["left_optimal"] is not None and b["right_optimal"] is not None:
        ax.axvspan(b["left_optimal"], b["right_optimal"], alpha=0.2, color="green", label="OPTIMAL зона")
    ax.axvspan(right_opt, right_marg, alpha=0.2, color="yellow", label="MARGINAL зона")
    ax.axvspan(right_marg, alpha_max, alpha=0.2, color="red", label="HOSTILE зона")

    ax.plot(data["alpha"], data["H"], "bo", markersize=6, label="Экспериментальные точки", alpha=0.6)
    ax.plot(alpha_fine, H_fine, "b-", linewidth=2, label=f"Модель (R²={model['r_squared']:.4f})")
    ax.plot(analysis["peak"]["alpha"], analysis["peak"]["H"], "g*", markersize=20, label="Пик модели")
    ax.plot(analysis["our"]["alpha"], analysis["our"]["H"], "r*", markersize=20, label="Наша Вселенная")

    for zone, thresh in THRESHOLDS.items():
        ax.axhline(y=thresh, color=ZONE_COLORS[zone], linestyle="--", alpha=0.5)

    ax.set_xlabel("α", fontsize=12)
    ax.set_ylabel("Индекс пригодности H", fontsize=12)
    ax.set_title("Данные и математическая модель", fontsize=14)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def create_visualization(data: dict[str, Any], model: dict[str, Any], analysis: dict[str, Any]) -> None:
    """Создает полную визуализацию с данными и моделью."""
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    alpha_min, alpha_max = STUDY_PARAMS["alpha_range"]
    alpha_fine = np.linspace(alpha_min, alpha_max, GRID_POINTS_PLOT)
    H_fine = model["poly"](alpha_fine)

    _plot_main_chart(axes[0, 0], data, model, analysis)

    # Остатки
    ax2 = axes[0, 1]
    residuals = data["H"] - model["poly"](data["alpha"])
    ax2.bar(range(len(residuals)), residuals, color="purple", alpha=0.6)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Точка данных", fontsize=12)
    ax2.set_ylabel("Остатки (H_data - H_model)", fontsize=12)
    ax2.set_title("Анализ остатков модели", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Производные
    ax3 = axes[1, 0]
    dH = np.gradient(H_fine, alpha_fine)
    ax3.plot(alpha_fine, dH, "r-", linewidth=2, label="dH/dα")
    ax3.axvline(x=analysis["our"]["alpha"], color="blue", linestyle="--", label="Наша α", alpha=0.5)
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax3.set_xlabel("α", fontsize=12)
    ax3.set_ylabel("Скорость изменения", fontsize=12)
    ax3.set_title("Производная модели", fontsize=14)
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    # Текстовый отчет
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.text(
        0.1, 0.95,
        _format_analysis_report(model, analysis),
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow"),
    )

    plt.suptitle("ПОЛНЫЙ АНАЛИЗ ПЛАТО ПРИГОДНОСТИ ВСЕЛЕННЫХ", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}/full_analysis_{_get_timestamp()}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n📸 График сохранен: {filename}")
    plt.show()

# ============================================================================
# 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================


def _analysis_to_serializable(analysis: dict[str, Any]) -> dict[str, Any]:
    """Преобразует анализ в структуру, пригодную для JSON."""
    boundaries = {
        k: float(v) if v is not None else None
        for k, v in analysis["boundaries"].items()
    }
    return {
        "peak": {k: float(v) for k, v in analysis["peak"].items()},
        "our": {k: float(v) if isinstance(v, (int, float)) else v for k, v in analysis["our"].items()},
        "derivatives": {k: float(v) for k, v in analysis["derivatives"].items()},
        "boundaries": boundaries,
        "integral": {k: float(v) for k, v in analysis["integral"].items()},
        "hostile_boundary": float(analysis["hostile_boundary"]),
    }


def save_results(data: dict[str, Any], model: dict[str, Any], analysis: dict[str, Any]) -> None:
    """Сохраняет все результаты в CSV, JSON и текстовый отчет."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = _get_timestamp()
    alpha_min, alpha_max = STUDY_PARAMS["alpha_range"]

    # CSV
    df = pd.DataFrame({"alpha": data["alpha"], "H": data["H"]})
    csv_file = f"{OUTPUT_DIR}/alpha_data_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"📊 Данные сохранены: {csv_file}")

    # JSON
    results_dict = {
        "timestamp": timestamp,
        "study_params": STUDY_PARAMS,
        "our_universe": OUR_UNIVERSE,
        "thresholds": THRESHOLDS,
        "model": {
            "coeffs": model["coeffs"].tolist(),
            "degree": model["degree"],
            "r_squared": model["r_squared"],
            "expression": model["expression"],
        },
        "analysis": _analysis_to_serializable(analysis),
    }
    json_file = f"{OUTPUT_DIR}/analysis_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"📋 Анализ сохранен: {json_file}")

    # Текстовый отчет
    txt_file = f"{OUTPUT_DIR}/report_{timestamp}.txt"
    report_body = _format_analysis_report(model, analysis)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЕТ ОБ ИССЛЕДОВАНИИ ПЛАТО ПРИГОДНОСТИ ВСЕЛЕННЫХ\n")
        f.write("=" * 80 + "\n\n")
        f.write("📊 ДАННЫЕ:\n")
        f.write(f"   Диапазон α: [{alpha_min}, {alpha_max}]\n")
        f.write(f"   Количество точек: {len(data['alpha'])}\n\n")
        f.write(report_body)
    print(f"📄 Отчет сохранен: {txt_file}")

# ============================================================================
# 7. ОСНОВНАЯ ПРОГРАММА
# ============================================================================


def main() -> None:
    """Запускает полный пайплайн анализа плато пригодности вселенных."""
    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА ПЛАТО ПРИГОДНОСТИ")
    print("=" * 80)

    data = generate_alpha_data()
    model = fit_polynomial(data["alpha"], data["H"], degree=3)
    analysis = analyze_model(model, data["alpha"], data["H"])
    create_visualization(data, model, analysis)
    save_results(data, model, analysis)

    print("\n" + "=" * 80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 80)


if __name__ == "__main__":
    main()
