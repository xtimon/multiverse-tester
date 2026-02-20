#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерактивное веб-демо MultiverseTester — «пузырь жизни».
Обёртка для обратной совместимости: загружает multiverse_tester.streamlit_demo.
Запуск: streamlit run streamlit_demo.py
      или: streamlit run -m multiverse_tester.streamlit_demo
      или: multiverse-demo (после pip install ".[demo]")
"""

import runpy
runpy.run_module("multiverse_tester.streamlit_demo", run_name="__main__")
