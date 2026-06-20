"""
Pytest 配置：把项目根目录加入 sys.path，避免不同启动方式下找不到 core。
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
