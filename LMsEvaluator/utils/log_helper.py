import logging
import os
import sys
import threading, time
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

class LogThreadManager:
    def __init__(self, logFilePath, socketio_instance, sid):
        self.logFilePath = logFilePath
        self.socketio = socketio_instance
        self.sid = sid
        self.stopWriteEvent = threading.Event() # 用于线程协作式停止的事件，即状态变量
        self.stopTailEvent = threading.Event() # 用于线程协作式停止的事件，即状态变量
        self.logWriteThread = None
        self.logTailThread = None

    def _simulate_log_writing(self):
        count = 0
        print(f"[{threading.current_thread().name}] 日志写入线程已启动")
        while not self.stopWriteEvent.is_set():
            with open(self.logFilePath, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]-【{count}】-INFO: This is a new message.\n")
            count += 1
            self.socketio.sleep(2) # 使用 socketio.sleep 兼容eventlet，让线程暂停2s
        print(f"[{threading.current_thread().name}] 日志写入线程即将退出")

    def _tail_log_file(self):
        try:
            print(f"[{threading.current_thread().name}] 日志监听线程已启动")
            # 初始时定位到文件末尾
            with open(self.logFilePath, 'r', encoding='utf-8') as f:
                f.seek(0, os.SEEK_END)
                last_pos = f.tell()

            while not self.stopTailEvent.is_set():
                self.socketio.sleep(1) # 每0.5秒检查一次新内容

                with open(self.logFilePath, 'r', encoding='utf-8') as f:
                    f.seek(last_pos)
                    new_content = f.read()
                    self.socketio.emit('new_log_entry', new_content, room=self.sid)
                    last_pos = f.tell()

        except Exception as e:
            print(f"[{threading.current_thread().name}] Error in tailing: {e}")
        finally:
            print(f"[{threading.current_thread().name}] 日志监听线程即将退出")
            
    def startWrite(self):	
        if (self.logWriteThread is None) or (not self.logWriteThread.is_alive()):
            self.stopWriteEvent.clear() # 清除信号，准备重新启动
            self.logWriteThread = threading.Thread(
                target=self._simulate_log_writing,
                name="logWriterThread" # 给线程命名方便调试
            )
            self.logWriteThread.daemon = True # 设置为守护线程，主程序退出时它也会退出
            self.logWriteThread.start()
            print("开始日志写入...")
        else:
            print("已存在写入线程")


    def startTail(self):
        if (self.logTailThread is None) or (not self.logTailThread.is_alive()):
            self.stopTailEvent.clear() # 确保停止信号已清除
            self.logTailThread = threading.Thread(
                target=self._tail_log_file,
                name="logTailThread" # 给线程命名方便调试
            )
            self.logTailThread.daemon = True
            self.logTailThread.start()
            print("开始监听日志...")
        else:
            print("已存在监听线程")

    def stopWrite(self):
        self.stopWriteEvent.set()
        if self.logWriteThread is not None and self.logWriteThread.is_alive():
            print("等待写入线程退出...")
            self.logWriteThread.join(timeout=5) # 等待线程结束，最多等待5秒
            if self.logWriteThread is not None and self.logWriteThread.is_alive():
                print("将强制退出...")

        self.logWriteThread = None
        print("监听线程退出成功")

    def stopTail(self):
        self.stopTailEvent.set() # 设置停止信号
        if self.logTailThread is not None and self.logTailThread.is_alive():
            print("等待监听线程退出...")
            self.logTailThread.join(timeout=5)
            if self.logTailThread is not None and self.logTailThread.is_alive():
                print("将强制退出...")

        self.logTailThread = None
        print("监听线程退出成功")