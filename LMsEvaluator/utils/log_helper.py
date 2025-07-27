import logging
import os
import sys
from datetime import datetime

# 全局变量存储当前文件处理器
current_file_handler = None


def logger_init(log_file_name='monitor',
                log_level=logging.INFO,
                log_dir='./logs/',
                only_file=False):
    global current_file_handler

    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建日志路径
    log_path = create_log_path(log_dir, log_file_name)
    print("Log file path:", log_path)

    # 创建格式化器
    formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建文件处理器
    current_file_handler = logging.FileHandler(log_path)
    current_file_handler.setFormatter(formatter)
    logger.addHandler(current_file_handler)

    # 如果不只是文件，添加控制台处理器
    if not only_file:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def create_log_path(log_dir, log_file_name):
    """创建日志文件路径"""
    return os.path.join(
        log_dir,
        f"{log_file_name}_{datetime.now().strftime('%Y-%m-%d')}.txt"
    )


def change_log_path(new_log_dir, new_log_file_name='monitor'):
    """动态修改日志文件路径"""
    global current_file_handler

    # 确保新目录存在
    if not os.path.exists(new_log_dir):
        os.makedirs(new_log_dir)

    # 创建新路径
    new_log_path = create_log_path(new_log_dir, new_log_file_name)
    print("New log file path:", new_log_path)

    # 获取根日志记录器
    logger = logging.getLogger()

    # 如果存在当前文件处理器，先关闭并移除
    if current_file_handler:
        # 关闭旧的文件处理器（重要！避免文件句柄泄漏）
        current_file_handler.close()
        logger.removeHandler(current_file_handler)

    # 创建新的文件处理器
    current_file_handler = logging.FileHandler(new_log_path)

    # 保持相同的格式
    formatter = logging.Formatter(
        '[%(asctime)s] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    current_file_handler.setFormatter(formatter)

    # 添加新的处理器
    logger.addHandler(current_file_handler)

    return new_log_path