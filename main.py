#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Точка входа для демонстрации MultiverseTester.
Запуск: python main.py

Author: Timur Isanov
Email: tisanov@yahoo.com
"""

import matplotlib.pyplot as plt

from multiverse_tester import (
    UniverseParameters,
    UniverseAnalyzer,
    MultiverseDynamicsExplorer,
)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 МУЛЬТИВСЕЛЕННЫЙ АНАЛИЗАТОР v3.0 (С РАСШИРЕННЫМ НУКЛЕОСИНТЕЗОМ)")
    print("=" * 60)

    explorer = MultiverseDynamicsExplorer()
    explorer.scan_parameter(
        param_name="alpha",
        start=1 / 500,
        stop=1 / 20,
        num_points=200,
        log_scale=False,
    )

    # График: индекс пригодности vs α
    result = explorer.results["alpha"]
    plt.figure(figsize=(10, 6))
    plt.plot(result["param_values"], result["habitability_scores"], "b.-", alpha=0.7)
    plt.axvline(x=ALPHA_OUR, color="r", linestyle="--", label="Наша Вселенная")
    plt.xlabel("α (постоянная тонкой структуры)")
    plt.ylabel("Индекс пригодности")
    plt.title("Зависимость пригодности от α")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    correlations = explorer.analyze_correlations("alpha")
    print("\n📊 Корреляции с α:")
    for prop, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"   {prop}: {corr:+.3f}")

    print("\n" + "=" * 60)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ НАШЕЙ ВСЕЛЕННОЙ (С РАСШИРЕННЫМ НУКЛЕОСИНТЕЗОМ)")
    print("=" * 60)

    our_analyzer = UniverseAnalyzer(UniverseParameters("🌍 Наша Вселенная"))
    index, score, metrics = our_analyzer.calculate_habitability_index()

    print(f"\n📊 Индекс пригодности: {score:.3f}")
    print(f"🏷️ Категория: {index.name}")
    print(f"\n📈 Метрики:")
    for metric, value in sorted(metrics.items()):
        print(f"   {metric}: {value:.2f}")

    nucleo = our_analyzer.stellar.complete_nucleosynthesis_analysis()
    print(f"\n🌟 ДЕТАЛЬНЫЙ НУКЛЕОСИНТЕЗ:")
    print(f"\n   🔥 Водородное горение:")
    print(f"      pp-цепочка: {nucleo['pp_chain']['rate_relative']:.2f} от солнечной")
    print(f"      CNO-цикл: {nucleo['cno_cycle']['rate_relative']:.2f} от солнечной")
    print(f"      Время жизни H: {nucleo['pp_chain']['tau_hydrogen_years']:.2e} лет")
    print(f"\n   ⚡ Гелиевое горение:")
    print(f"      Тройная α: {nucleo['triple_alpha']['rate_relative']:.2f} от солнечной")
    print(f"      Резонанс углерода: {nucleo['triple_alpha']['resonance_energy_kev']:.1f} кэВ")
    print(f"      Производство C: {nucleo['triple_alpha']['carbon_production']}")
    print(f"\n   💫 Альфа-процесс (от Ne до Fe):")
    for r in nucleo['alpha_process'][:5]:
        print(f"      {r['nucleus']}: относительный выход {r['relative_yield']:.3f}")
    print(f"\n   🌌 Нейтронные процессы:")
    print(f"      s-процесс: {nucleo['s_process']['path']}")
    print(f"      r-процесс: трансурановых элементов {nucleo['r_process']['transuranic_elements']}")
    print(f"\n   💥 Сверхновые:")
    print(f"      Масса Fe ядра: {nucleo['supernova']['fe_core_mass']:.2f} M☉")
    print(f"      Тип коллапса: {nucleo['supernova']['collapse_type']}")
    print(f"      r-процесс возможен: {nucleo['supernova']['r_process_possible']}")

    print("\n" + "=" * 60)
    print("🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 60)
