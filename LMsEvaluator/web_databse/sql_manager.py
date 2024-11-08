import sqlite3
from werkzeug.security import generate_password_hash

# 创建数据库和用户表
def create_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender INTEGER NOT NULL,
        permissions TEXT
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
        print(f"ID: {user[0]}, Username: {user[1]}, Role: {user[3]}, Age: {user[4]}, Gender: {user[5]}, Permissions: {user[6]}")
    conn.close()

# 添加用户
def add_user(username, password, role, age, gender, permissions):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    hashed_password = generate_password_hash(password)  # 哈希密码
    try:
        cursor.execute('INSERT INTO user (username, password, role, age, gender, permissions) VALUES (?, ?, ?, ?, ?, ?)',
                       (username, hashed_password, role, age, gender, permissions))
        conn.commit()
        print(f"User '{username}' added successfully.")
    except sqlite3.IntegrityError:
        print(f"User '{username}' already exists.")
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

# 运行示例
if __name__ == '__main__':
    create_database()
    # 示例用户操作
    add_user('zcy', '123', 'user', 25, 0, "'edit', 'delete', 'add'")  # 示例用户
    add_user('admin', '888888', 'admin', 30, 1, "'edit', 'delete', 'add'")  # 示例管理员用户
    print_users()
