"""把项目根目录加入 sys.path，使脚本既能用 `python scripts/xxx.py` 直接运行，
也能用 `python -m scripts.xxx` 运行。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
