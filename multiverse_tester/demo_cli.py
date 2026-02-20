#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск Streamlit-демо MultiverseTester.
Использует streamlit run для интерактивного веб-приложения.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Запускает streamlit run с путём к streamlit_demo.py."""
    demo_path = Path(__file__).parent / "streamlit_demo.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(demo_path),
        *sys.argv[1:],
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
