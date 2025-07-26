import sqlite3
import yaml
from werkzeug.security import generate_password_hash
from datetime import datetime


# 添加数据用
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.database_helper import extractResult
lmsDir = os.path.dirname(os.path.abspath(__file__))
# 使用os.path.join来正确拼接路径
data_path = os.path.join(lmsDir, 'users.db')
# 创建数据库和表
def create_database():
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()

    # 创建用户表，添加 email 字段
    cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender INTEGER NOT NULL,
        permissions TEXT,
        token TEXT,
        token_refresh TEXT,
        login_time DATETIME,
        avatar_url TEXT,
        email TEXT UNIQUE NOT NULL  -- 新增邮箱字段
    )
    ''')

    # 创建验证码表
    cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS verification_code (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        code TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 创建攻击记录表，createUserName 字段改为 TEXT 类型以匹配 User.username
    cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS attack_record (
        attackID INTEGER PRIMARY KEY AUTOINCREMENT,
				attackName TEXT NOT NULL,
        createUserName TEXT NOT NULL,
        createTime BIGINT DEFAULT (CAST(strftime('%s', 'now') AS BIGINT)) NOT NULL,
        isTreasure BOOLEAN DEFAULT FALSE,
				attackInfo TEXT,  
        attackResult TEXT,
        FOREIGN KEY (createUserName) REFERENCES user (username)
    )
    ''')

    conn.commit()
    conn.close()


# 添加用户
def add_user(username, password, role, age, gender, permissions, email, avatar_url=None):
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    hashed_password = generate_password_hash(password)  # 哈希密码
    try:
        cursor.execute(
            'INSERT INTO user (username, password, role, age, gender, permissions, email, avatar_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (username, hashed_password, role, age, gender, permissions, email, avatar_url))
        conn.commit()
        print(f"User '{username}' added successfully.")
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists.")
    conn.close()


# 更新用户 Token 和登录时间
def update_user_token(username, token):
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    login_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间
    cursor.execute('UPDATE user SET token = ?, login_time = ? WHERE username = ?',
                   (token, login_time, username))
    conn.commit()
    if cursor.rowcount > 0:
        print(f"Token and login time updated for user '{username}'.")
    else:
        print(f"User '{username}' not found.")
    conn.close()


# 删除用户
def delete_user(username):
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM user WHERE username = ?', (username,))
    conn.commit()
    if cursor.rowcount > 0:
        print(f"User '{username}' deleted successfully.")
    else:
        print(f"User '{username}' not found.")
    conn.close()


# 添加验证码
def add_verification_code(email, code):
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO verification_code (email, code) VALUES (?, ?)', (email, code))
    conn.commit()
    print(f"Verification code for '{email}' added successfully.")
    conn.close()


# 打印用户数据
def print_users():
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user')
    users = cursor.fetchall()
    for user in users:
        print(f"ID: {user[0]}, Username: {user[1]}, Role: {user[3]}, Age: {user[4]}, Gender: {user[5]}, "
              f"Permissions: {user[6]}, Token: {user[7]}, Login Time: {user[8]}, Avatar URL: {user[9]}, Email: {user[10]}")
    conn.close()


# 打印验证码
def print_verification_codes():
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM verification_code')
    codes = cursor.fetchall()
    for code in codes:
        print(f"ID: {code[0]}, Email: {code[1]}, Code: {code[2]}, Created At: {code[3]}")
    conn.close()


# 添加攻击记录
def add_attack_record(attackName, createUserName, createTime, attackResult, attackInfo=None, isTreasure=False):
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO attack_record (attackName, createUserName, createTime, isTreasure, attackInfo, attackResult) VALUES (?, ?, ?, ?, ?, ?)',
                   (attackName, createUserName, createTime, isTreasure, json.dumps(attackInfo), str(attackResult)))
    conn.commit()
    print(f"Attack record added for user '{createUserName}'.")
    conn.close()


# 打印攻击记录
def print_attack_records():
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM attack_record')
    records = cursor.fetchall()
    for record in records:
        print(f"AttackID: {record[0]}, createUserName: {record[1]}, CreateTime: {record[2]}, AttackResult: {record[3]}")
    conn.close()


# 添加日志记录
def add_log_record(username, log_filename, log_status):
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO log_record (username, log_filename, log_status) VALUES (?, ?, ?)',
                   (username, log_filename, log_status))
    conn.commit()
    print(f"Log record for user '{username}' added successfully.")
    conn.close()


# 打印日志记录
def print_log_records():
    conn = sqlite3.connect(data_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM log_record')
    logs = cursor.fetchall()
    for log in logs:
        print(f"LogID: {log[0]}, Username: {log[1]}, Log Filename: {log[2]}, Status: {log[3]}, Log Time: {log[4]}")
    conn.close()
    
# 解析配置信息
def get_attack_info(username):
    output_dir = os.path.dirname(os.path.abspath(__file__))
    info_file = os.path.join(output_dir, "test_data", f"{username}_attack_info.yaml")
    if not os.path.exists(info_file):
        print(f"攻击信息文件不存在: {info_file}")
        return []
    try:
        with open(info_file, 'r' ,encoding='gb2312') as file:
            attack_info = yaml.safe_load(file)
            return attack_info if attack_info else []
    except Exception as e:
        print(f"读取攻击信息文件失败: {e}")
        return []


# 运行示例
if __name__ == '__main__':
    print("找到数据库：", data_path)
    create_database()
		
    # add_user('admin', '888888', 'admin', 30, 1, "'edit', 'delete', 'add'", 'admin@example.com', 'https://gitee.com/topiza/image/raw/master/file_3.png')
    # # 示例用户操作，带邮箱
    # add_user('zcy', '123', 'user', 25, 0, "'edit', 'delete', 'add'", 'zcy@example.com', 'https://gitee.com/topiza/image/raw/master/file_1.png')
    # add_user('admin', '888888', 'admin', 30, 1, "'edit', 'delete', 'add'", 'admin@example.com', 'https://gitee.com/topiza/image/raw/master/file_3.png')
    # update_user_token('zcy', 'sample_token')
    # print_users()

    # # 示例验证码操作
    # add_verification_code('example@example.com', '123456')
    # print_verification_codes()

    # # 示例攻击记录操作
    # add_attack_record('zcy', {"result": "success"})
    # print_attack_records()
    
    # 向数据库中添加攻击记录
    lmsDir = os.path.dirname(os.path.abspath(__file__))
    filename = "u1h_single_1737727113_2025-01-24.txt"
    info = filename.split('_')
    username = 'ChenyangZhao'
    initTime = eval(info[2])
    result = extractResult(os.path.join(lmsDir, "test_data",filename))
    add_attack_record('ForDEBUG', username, initTime, json.dumps(result), attackInfo=get_attack_info('admin'), isTreasure=False)
    

    # # 示例日志操作
    # add_log_record('zcy', 'admin_single_1737092485_2024-12-04.txt', 'FINISHED')
    # print_log_records()
