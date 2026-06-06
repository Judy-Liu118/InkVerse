"""
core.logger -- 统一日志系统

替换全局 print()，支持控制台 + 文件双输出。
每次启动创建带时间戳的日志文件，保留最近 10 个。
"""
import os
import sys
import glob
import logging
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_ROOT, "outputs", "log")
_initialized = False


def setup_logging(level: int = logging.DEBUG, log_file: str = None) -> None:
    """初始化全局日志配置（幂等，多次调用不重复注册）。"""
    global _initialized
    if _initialized:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    if log_file:
        path = log_file
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_LOG_DIR, f"inkverse_{ts}.log")

    root_logger = logging.getLogger("inkverse")
    root_logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # 文件 handler
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    _initialized = True
    root_logger.info("日志系统初始化完成 | 文件=%s", path)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger，自动继承 inkverse 根 logger 的 handlers。
    若 setup_logging 尚未调用，先用默认参数懒初始化，确保日志不丢失。"""
    if not _initialized:
        setup_logging()
    return logging.getLogger(f"inkverse.{name}")
