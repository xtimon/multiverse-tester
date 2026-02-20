#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск Streamlit-демо MultiverseTester.
Использует streamlit run для интерактивного веб-приложения.
"""

import sys
import subprocess


def main():
    """Запускает streamlit run -m multiverse_tester.streamlit_demo с переданными аргументами."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "-m",
        "multiverse_tester.streamlit_demo",
        *sys.argv[1:],
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
