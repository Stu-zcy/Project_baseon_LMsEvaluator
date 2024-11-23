import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime


# 创建数据库和表
def create_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # 创建用户表
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
        login_time DATETIME
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

    # 创建攻击记录表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attack_record (
        attackID INTEGER PRIMARY KEY AUTOINCREMENT,
        createUserID INTEGER,
        createTime DATETIME DEFAULT CURRENT_TIMESTAMP,
        attackResult TEXT,
        FOREIGN KEY (createUserID) REFERENCES user (id)
    )
    ''')

    conn.commit()
    conn.close()


# 打印用户数据
def print_users():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user')
    users = cursor.fetchall()
    for user in users:
        print(f"ID: {user[0]}, Username: {user[1]}, Role: {user[3]}, Age: {user[4]}, Gender: {user[5]}, "
              f"Permissions: {user[6]}, Token: {user[7]}, Login Time: {user[8]}")
    conn.close()


# 添加用户
def add_user(username, password, role, age, gender, permissions):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    hashed_password = generate_password_hash(password)  # 哈希密码
    try:
        cursor.execute(
            'INSERT INTO user (username, password, role, age, gender, permissions) VALUES (?, ?, ?, ?, ?, ?)',
            (username, hashed_password, role, age, gender, permissions))
        conn.commit()
        print(f"User '{username}' added successfully.")
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists.")
    conn.close()


# 更新用户 Token 和登录时间
def update_user_token(username, token):
    conn = sqlite3.connect('users.db')
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
    conn = sqlite3.connect('users.db')
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
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO verification_code (email, code) VALUES (?, ?)', (email, code))
    conn.commit()
    print(f"Verification code for '{email}' added successfully.")
    conn.close()


# 打印验证码
def print_verification_codes():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM verification_code')
    codes = cursor.fetchall()
    for code in codes:
        print(f"ID: {code[0]}, Email: {code[1]}, Code: {code[2]}, Created At: {code[3]}")
    conn.close()


# 添加攻击记录
def add_attack_record(createUserID, attackResult):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO attack_record (createUserID, attackResult) VALUES (?, ?)',
                   (createUserID, str(attackResult)))
    conn.commit()
    print(f"Attack record added for user ID '{createUserID}'.")
    conn.close()


# 打印攻击记录
def print_attack_records():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM attack_record')
    records = cursor.fetchall()
    for record in records:
        print(f"AttackID: {record[0]}, CreateUserID: {record[1]}, CreateTime: {record[2]}, AttackResult: {record[3]}")
    conn.close()


# 运行示例
if __name__ == '__main__':
    create_database()

    # 示例用户操作
    add_user('zcy', '123', 'user', 25, 0, "'edit', 'delete', 'add'")
    add_user('admin', '888888', 'admin', 30, 1, "'edit', 'delete', 'add'")
    update_user_token('zcy', 'sample_token')
    print_users()

    # 示例验证码操作
    add_verification_code('example@example.com', '123456')
    print_verification_codes()

    # 示例攻击记录操作
    add_attack_record(1, {"result": "success"})
    print_attack_records()
