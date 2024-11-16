from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
import os
import json
from utils.config_parser import parse_config
from werkzeug.security import generate_password_hash, check_password_hash
import random,time
import string
from datetime import datetime, timedelta
import extract
from jwt_token import sign

app = Flask(__name__)
CORS(app)

# 邮箱配置
app.config['MAIL_DEBUG'] = True
app.config['MAIL_SUPPRESS_SEND'] = False
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '2544073891@qq.com'
app.config['MAIL_PASSWORD'] = 'jblpfwtrutloecai'  # 授权码
app.config['MAIL_DEFAULT_SENDER'] = '2544073891@qq.com'

mail = Mail(app)

# 数据库配置
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:\\Desktop\\Project\\LMsEvaluator\\web_databse\\users.db'
db = SQLAlchemy(app)

# 用户模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    permissions = db.Column(db.String(200), nullable=True)

# 验证码模型
class VerificationCode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    code = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@app.before_request
def create_tables():
    db.create_all()
    print("created!")



@app.route('/api/send_verification_code', methods=['POST'])
def send_verification_code():
    email = request.json.get('email')

    # 验证邮箱格式
    if not email or '@' not in email:
        return jsonify({"message": "无效的邮箱地址"}), 400

    # 生成验证码，排除易混淆字符
    characters = string.ascii_letters + string.digits
    characters = characters.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
    verification_code = ''.join(random.choices(characters, k=6))

    # 检查数据库中是否已经有该邮箱的验证码
    existing_code = VerificationCode.query.filter_by(email=email).first()

    if existing_code:
        # 如果已有验证码，且未过期，则更新验证码和时间戳
        if datetime.utcnow() - existing_code.created_at < timedelta(minutes=5):
            existing_code.code = verification_code
            existing_code.created_at = datetime.utcnow()
            db.session.commit()
            message = "验证码已更新"
        else:
            # 如果验证码已过期，删除旧的记录，生成新的验证码
            db.session.delete(existing_code)
            db.session.commit()
            new_code = VerificationCode(email=email, code=verification_code, created_at=datetime.utcnow())
            db.session.add(new_code)
            db.session.commit()
            message = "验证码已发送"
    else:
        # 如果没有该邮箱的验证码，直接创建新记录
        new_code = VerificationCode(email=email, code=verification_code, created_at=datetime.utcnow())
        db.session.add(new_code)
        db.session.commit()
        message = "验证码已发送"

    # 发送邮件
    msg = Message('您的验证码', recipients=[email])
    msg.body = f'您的验证码是: {verification_code}，有效时间为5分钟。'
    try:
        mail.send(msg)
    except Exception as e:
        return jsonify({"message": f"发送验证码失败: {str(e)}"}), 500

    return jsonify({"message": message, "code": verification_code}), 200


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data['username']
    password = data['password']
    age = 0
    gender = 1  # 如果没有前端提供性别，默认为1（男）
    verification_code = data.get('verificationCode')
    email = data.get('email')  # 前端传递邮箱地址

    # 验证验证码逻辑
    if not verification_code:
        return jsonify({"message": "验证码不能为空"}), 400

    code_entry = VerificationCode.query.filter_by(email=email, code=verification_code).first()
    if not code_entry:
        return jsonify({"message": "验证码无效"}), 400

    # 检查验证码是否过期（5分钟有效期）
    if datetime.utcnow() - code_entry.created_at > timedelta(minutes=5):
        return jsonify({"message": "验证码已过期"}), 400

    # 添加用户
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password, role='user', age=age, gender=gender,permissions="'edit', 'delete', 'add'")
    db.session.add(new_user)
    db.session.commit()

    # 删除已经使用过的验证码（可选，避免数据库积累过多无效数据）
    db.session.delete(code_entry)
    db.session.commit()

    return jsonify({"message": "注册成功"}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password']

    print(f"Trying to login with username: {username}")

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        account_info = {
            'role': user.role,
            'age': user.age,
            'gender': user.gender,
            'permissions': user.permissions.split(',') if user.permissions else []
        }
        expires_in = 60 * 60 * 1000  # 设置过期时间为1小时
        jwt_token = sign({"username": username, "role": user.role}, 'secret key', expires_in)

        # 返回数据与前端 Mock 格式保持一致
        return jsonify({
            "code": 0,
            "message": "success",
            "data": {
                "token": jwt_token,
                "expires": expires_in + int(time.time() * 1000)  # 当前时间 + 过期时间
            },
            "accountInfo": account_info
        }), 200

    return jsonify({
        "code": 401,
        "message": "用户名或密码错误"
    }), 401


@app.route('/api/attack_List', methods=['POST'])
def receive_attack_List():
    data = request.json
    attack_list = data.get('attack_list', [])
    username = data.get('username', None)

    if username:
        print(f"Received data from user: {username}")
    else:
        return jsonify({'status': 'error', 'message': 'Username is missing!'}), 400

    print("Received attack list:", attack_list)  # 输出接收到的数据


    return jsonify({'status': 'success', 'message': 'Attack list received!', 'received_data': attack_list})


@app.route('/api/execute_attack', methods=['POST'])
def execute_attack():
    try:
        data = request.json
        username = data.get('username', None)  # 获取用户名

        if not username:
            return jsonify({'status': 'error', 'message': 'Username is missing!'}), 400

        print(f"Executing attack for user: {username}")

        # 下游任务执行代码
        project_path = os.path.dirname(os.path.abspath(__file__))
        model_class = parse_config(project_path)
        model_class.run()

        return jsonify({'status': 'success', 'message': 'Attack executed successfully!'})

    except Exception as e:
        print("Error executing attack:", e)
        return jsonify({'status': 'error', 'message': 'Failed to execute attack'}), 500

@app.route('/api/defense_list', methods=['POST'])
def receive_defense_list():
    data = request.json
    attack_list = data.get('attack_list', [])
    username = data.get('username', None)

    if username:
        print(f"Received data from user: {username}")
    else:
        return jsonify({'status': 'error', 'message': 'Username is missing!'}), 400

    print("Received attack list:", attack_list)  # 输出接收到的数据


    return jsonify({'status': 'success', 'message': 'Attack list received!', 'received_data': attack_list})


@app.route('/api/execute_defense', methods=['POST'])
def execute_defense():
    try:
        data = request.json
        username = data.get('username', None)  # 获取用户名

        if not username:
            return jsonify({'status': 'error', 'message': 'Username is missing!'}), 400

        print(f"Executing attack for user: {username}")

        # 下游任务执行代码
        project_path = os.path.dirname(os.path.abspath(__file__))
        model_class = parse_config(project_path)
        model_class.run()

        return jsonify({'status': 'success', 'message': 'Attack executed successfully!'})

    except Exception as e:
        print("Error executing attack:", e)
        return jsonify({'status': 'error', 'message': 'Failed to execute attack'}), 500

@app.route('/api/submit_log', methods=['POST'])
def submit_log():
    try:
        path = "./logs/single_2024-04-25.txt"  # 日志文件路径,之后改成自动获取
        res = json.dumps(extract.extractResult(path), indent=2)  # 提取日志内容
        return jsonify(res)  # 返回 JSON 格式的日志内容
    except Exception as e:
        print(f"Error fetching log: {e}")
        return jsonify({"error": "Failed to fetch log", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000,debug=True)
