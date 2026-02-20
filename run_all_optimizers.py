#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакетный запуск всех оптимизаторов MultiverseTester.
Перенаправляет на multiverse_tester.run_all_optimizers.
Запуск: python run_all_optimizers.py
      или: python -m multiverse_tester.run_all_optimizers
      или: multiverse-run-optimizers (после pip install .)
"""

from multiverse_tester.run_all_optimizers import main

if __name__ == "__main__":
    main()
