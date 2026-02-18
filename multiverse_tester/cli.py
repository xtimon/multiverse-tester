#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry point for MultiverseTester.

Author: Timur Isanov
Email: tisanov@yahoo.com
"""


def main():
    """Run main multiverse analysis demo."""
    from multiverse_tester import (
        UniverseParameters,
        UniverseAnalyzer,
        MultiverseDynamicsExplorer,
    )

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

    correlations = explorer.analyze_correlations("alpha")
    print("\n📊 Корреляции с α:")
    for prop, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"   {prop}: {corr:+.3f}")

    print("\n" + "=" * 60)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ НАШЕЙ ВСЕЛЕННОЙ")
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
    print(f"   pp-цепочка: {nucleo['pp_chain']['rate_relative']:.2f}")
    print(f"   Тройная α: {nucleo['triple_alpha']['rate_relative']:.2f}")
    print(f"   Тип коллапса: {nucleo['supernova']['collapse_type']}")

    print("\n" + "=" * 60)
    print("🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 60)


if __name__ == "__main__":
    main()
