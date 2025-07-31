import os
import time
import json
import traceback
from datetime import datetime

from test4transformers import run_pipeline
from user_config.config_gen import get_attack_info, update_log_file_name
from utils import extractResult
from web_databse.sql_manager import add_attack_record, update_attack_result


def run_attack_thread(username, attackName):
    initTime = int(time.time())  # 提前定义，保证异常时也能访问
    try:
        # 1. 获取攻击配置信息
        attack_info = get_attack_info(username)
        if not attack_info or not attack_info[0]:
            raise ValueError(f"未获取到攻击信息，username={username}")

        attackProgress = f"0/{len(attack_info[0])}"
        date = str(datetime.now())[:10]

        # 2. 添加攻击记录
        add_attack_record(
            attackName=attackName,
            createUserName=username,
            createTime=initTime,
            attackInfo=attack_info,
            attackResult=json.dumps("RUNNING"),
            attackProgress=attackProgress,
            reportState=0
        )

        # 3. 更新日志文件名并执行攻击
        project_path = os.path.dirname(os.path.abspath(__file__))
        fileName = f"{username}_single_{initTime}"
        update_log_file_name(username, fileName)

        config_path = os.path.join(project_path, 'user_config', f"{username}_config.yaml")
        run_pipeline(config_path)

        # 4. 提取攻击结果
        log_file = os.path.join(project_path, 'logs', f"{fileName}_{date}.txt")
        result = extractResult(log_file)

        # 5. 更新攻击结果
        update_attack_result( username,initTime, json.dumps(result))
        print(f"[{username}] 攻击成功，结果已更新。")

    except Exception as e:
        print(f"[{username}] 后台线程执行失败：", e)
        traceback.print_exc()  # 打印堆栈方便调试
        update_attack_result(initTime, username, json.dumps("FAILED"))