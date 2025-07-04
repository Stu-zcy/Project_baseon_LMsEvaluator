from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from functools import wraps
import jwt
import os
import json
from utils.config_parser import parse_config
from werkzeug.security import generate_password_hash, check_password_hash
import random, time
import string
from datetime import datetime, timedelta
from utils.database_helper import extractResult
from jwt_token import sign
from user_config.config_gen import generate_config

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
expires_in = 60 * 60 * 1000
expires_in_refresh = 3 * expires_in
mail = Mail(app)
project_path = os.path.dirname(os.path.abspath(__file__))
# 数据库配置
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:\\Desktop\\Project\\LMsEvaluator\\web_databse\\users.db'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:\\Users\\yjh\\Desktop\\Project_baseon_LMsEvaluator\\LMsEvaluator\\web_databse\\users.db'
lmsDir = os.path.dirname(os.path.abspath(__file__))
# 使用os.path.join来正确拼接路径
data_path = os.path.join(lmsDir, 'web_databse', 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' +f"{data_path}"
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(80), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    permissions = db.Column(db.String(200), nullable=True)
    token = db.Column(db.String(500), nullable=True)  # 存储登录生成的Token
    token_refresh = db.Column(db.String(500), nullable=True)
    login_time = db.Column(db.DateTime, nullable=True)  # 上次登录时间
    avatar_url = db.Column(db.String(500), nullable=True)  # 存储头像链接，最大500个字符
    email = db.Column(db.String(120), unique=True, nullable=False)  # 新增邮箱字段，唯一且不能为空


# 验证码模型
class VerificationCode(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), nullable=False)
    code = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AttackRecord(db.Model):
    attackID = db.Column(db.Integer, autoincrement=True, primary_key=True)
    createUserName = db.Column(db.String, db.ForeignKey(User.username), nullable=False)
    createTime = db.Column(db.BigInteger, default=int(time.time()), nullable=False)
    attackResult = db.Column(db.String, nullable=False)
    isTreasure = db.Column(db.Boolean, default=False)


# def afterInsertListener_AttackRecord

@app.before_request
def create_tables():
    db.create_all()
    print("created!")

def verify_token(token, username):

  # 解码 token
  # 检查 token 是否匹配
  user = User.query.filter_by(username=username).first()
  if (not user) or (user.token != token):
    raise jwt.InvalidSignatureError

  # 检查 token 是否过期
  login_time = user.login_time
  if not login_time or datetime.utcnow() > login_time + timedelta(hours=1):
    raise jwt.ExpiredSignatureError
  
def verify_token_refresh(token_refresh, username):
  # 解码 token
  # 检查 token 是否匹配
  user = User.query.filter_by(username=username).first()
  if (not user) or (user.token_refresh != token_refresh):
    raise jwt.InvalidSignatureError
  
  # 检查 token 是否过期
  login_time = user.login_time
  if not login_time or datetime.utcnow() > login_time + timedelta(hours=3):
    raise jwt.ExpiredSignatureError

def auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        username = request.json.get('username', None)
        
        # 从请求头获取token
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Bearer token
            except IndexError:
                return jsonify({
                    'code': 401,
                    'message': '无效的token格式'
                }), 401

        if not (token and username):
            return jsonify({
                'code': 401,
                'message': '缺少token或用户名'
            }), 401

        try:  
            verify_token(token, username)
        except jwt.ExpiredSignatureError:
            return jsonify({
                'code': 403,
                'message': 'token已过期'
            }), 403
        except jwt.InvalidSignatureError:
            return jsonify({
                'code': 401,
                'message': '无效的token或用户名'
            }), 401

        return f(*args, **kwargs)
    return decorated

@app.route('/api/refresh_token', methods=['POST'])
def refresh_token():
    refresh_token = request.json.get('refresh_token', None)
    username = request.json.get('username', None)
    if not (refresh_token and username):
        return jsonify({
            'code': 401,
            'message': '无效的token或用户名'
        }), 401

    try:
        verify_token_refresh(refresh_token, username)
    except jwt.ExpiredSignatureError:
        return jsonify({
            'code': 403,
            'message': 'token已过期'
        }), 403
    except jwt.InvalidSignatureError:
        return jsonify({
            'code': 401,
            'message': '无效的token或用户名'
        }), 401

    try: 
        user = User.query.filter_by(username=username).first()
        jwt_token = sign({"id": user.id, "username": username, "role": user.role}, 'secret key', expires_in)
        jwt_token_refresh = sign({"id": user.id, "username": username, "role": user.role}, 'secret key', expires_in_refresh)

        # 更新数据库中的 token 和 login_time
        user.token = jwt_token
        user.token_refresh = jwt_token_refresh
        user.login_time = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'code': 200,
            'message': 'success',
            'data': {
                'token': jwt_token,
                'token_refresh': jwt_token_refresh
            }
        }), 200
    except Exception:
        return jsonify({
            'code': 500,
            'message': 'failed to get a new token'
        }), 500

  

@app.route('/api/send_verification_code', methods=['POST'])
def send_verification_code():
    email = request.json.get('email')

    # 验证邮箱格式
    if not email or '@' not in email:
        return jsonify({
            "code": 400,
            "message": "无效的邮箱地址"
        }), 400

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
        return jsonify({
            "code": 500,
            "message": f"发送验证码失败: {str(e)}"
        }), 500

    return jsonify({
        "code": 200,
        "message": message, 
        "data": {
            'code': verification_code
        }
    }), 200


@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data['username']
    password = data['password']
    avatar_url = data['avatar']
    verification_code = data['verificationCode']
    email = data['email']
    age = 0
    gender = 1  # 如果没有前端提供性别，默认为1（男）

    # 验证验证码逻辑
    if not verification_code:
        return jsonify({
            "code": 400,
            "message": "验证码不能为空"
        }), 400

    code_entry = VerificationCode.query.filter_by(email=email, code=verification_code).first()
    if not code_entry:
        return jsonify({
            "code": 400,
            "message": "验证码无效"
        }), 400

    # 检查验证码是否过期（5分钟有效期）
    if datetime.utcnow() - code_entry.created_at > timedelta(minutes=5):
        return jsonify({
            "code": 400,
            "message": "验证码已过期"
        }), 400

    # 添加用户
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password, role='user', age=age, gender=gender,
                    permissions="'edit', 'delete', 'add'", avatar_url=avatar_url, email=email)

    db.session.add(new_user)
    db.session.commit()

    # 删除已经使用过的验证码（可选，避免数据库积累过多无效数据）
    db.session.delete(code_entry)
    db.session.commit()

    return jsonify({
        "code": 201,
        "message": "注册成功"
    }), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password']

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        account_info = {
            'role': user.role,
            'age': user.age,
            'gender': user.gender,
            'permissions': user.permissions.strip(',') if user.permissions else [],
            'avatar_url': user.avatar_url,
        }

        jwt_token = sign({"id": user.id, "username": username, "role": user.role}, 'secret key', expires_in)
        #刷新用的token，这个token过期就重新登录，没过期就重新申请一个
        jwt_token_refresh = sign({"id": user.id, "username": username, "role": user.role}, 'secret key', expires_in_refresh)

        # 更新数据库中的 token 和 login_time
        user.token = jwt_token
        user.token_refresh = jwt_token_refresh
        user.login_time = datetime.utcnow()
        db.session.commit()

        return jsonify({
            "code": 200,
            "message": "success",
            "data": {
                "token": jwt_token,
                # "expires": expires_in * 1000 + int(time.time() * 1000)  # 当前时间 + 过期时间
                "token_refresh": jwt_token_refresh
            },
            "accountInfo": account_info
        }), 200

    return jsonify({
        "code": 401,
        "message": "用户名或密码错误"
    }), 401

#存储在account.ts的profile函数中：
@app.route('/api/profile', methods=['POST'])
@auth
def profile():
    try:
        data = request.json
        username = data.get('username', None).strip('"')

        user = User.query.filter_by(username=username).first()
        account = {'username': user.username,'avatar': user.avatar_url,'gender': user.gender, 'age': user.age}
        data={'account':account,'permissions': user.permissions.strip(',') if user.permissions else [],'role': user.role}
        print(data)
        return jsonify({
            'account':account,
            'permissions': user.permissions.strip(',') if user.permissions else [],
            'role': user.role
        }), 200

    except Exception as e:
        print("Error executing attack:", e)
        return jsonify({}), 500


# 验证用户身份
# @app.route('/api/auth', methods=['POST'])
# def auth():
#     try:
#         data = request.json
#         username = data.get('username', None).strip('"')
#         token = data.get('token', None).strip('"')

#         # 验证用户名和 token
#         if not username or not token:
#             return jsonify({'status': 'error', 'message': 'Username or token is missing!'}), 400

#         # 验证token的有效性
#         is_valid, message = verify_token(token, username)
#         if not is_valid:
#             return jsonify({'status': 'error', 'message': message}), 402
#         print("token successfully verified")
#         # 获取用户信息
#         user = User.query.filter_by(username=username).first()
#         print(user)
#         if not user:
#             return jsonify({'status': 'error', 'message': 'User not found'}), 404

#         # 验证是否为admin
#         if user.role != 'admin':
#             return jsonify({'status': 'error', 'message': 'Not admin'}), 401

#         return jsonify({'status': 'success', 'message': 'Authenticated'}), 200

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': 'Failed to fetch auth', 'error': str(e)}), 500


# 获取用户列表，只返回role为'user'的用户
@app.route('/api/users', methods=['POST'])
@auth
def get_users():
    try:
        data = request.json
        username = data.get('username', None).strip('"')
        # 获取用户信息
        user = User.query.filter_by(username=username).first()
        # 验证是否为admin
        if user.role != 'admin':
            return jsonify({'status': 'error', 'message': 'Not admin'}), 401

        # 获取所有role为'user'的用户
        users = User.query.filter_by(role='user').all()
        user_data = []
        for user in users:
            user_info = {
                'username': user.username,
                'avatar_url': user.avatar_url,
                'age': user.age,
                'gender': user.gender,
                'email': user.email,

            }
            user_data.append(user_info)
            print(user_data)

        return jsonify({'status': 'success', 'users': user_data}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 删除用户
@app.route('/api/delete_users', methods=['POST'])
@auth
def delete_user():
    try:
        data = request.json
        username = data.get('username', None).strip('"')
        delete_username = data.get('delete_username', None).strip('"')


        # 获取请求用户信息
        user = User.query.filter_by(username=username).first()
        # 验证是否为admin
        if user.role != 'admin':
            return jsonify({'status': 'error', 'message': 'Not admin'}), 401

        # 获取要删除的用户
        delete_user = User.query.filter_by(username=delete_username).first()

        if not delete_user:
            return jsonify({'status': 'error', 'message': 'User to delete not found'}), 404

        # 删除用户
        db.session.delete(delete_user)
        db.session.commit()

        return jsonify({'status': 'success', 'message': 'User deleted successfully'}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/updateInfo', methods=['POST'])
@auth
def updateProfile():
    data = request.json
    username = data.get('username', None).strip('"')
    newUsername = data.get('newUsername', None)
    newAvatar = data.get('newAvatar', None)
    newGender = data.get('newGender', None)
    newAge = data.get('newAge', None)
    
    # TODO: 检测用户名是否重复
    count = User.query.filter_by(username=username).update({
        User.username: newUsername,
        User.avatar_url: newAvatar,
        User.gender: newGender,
        User.age: newAge
    })
    db.session.commit()
    if count:
        return jsonify({'status': 'success', 'message': 'success to update'}), 200
    else:
        return jsonify({'status': 'error', 'message': 'no users match'}), 500


@app.route('/api/attack_list', methods=['POST'])
@auth
def receive_attack_list():
    data = request.json
    attack_list = data.get('attack_list', [])
    username = data.get('username', None).strip('"')

    print(f"Received data from user: {username}")
    print("Received attack list:", attack_list)  # 输出接收到的数据
    generate_config(username, attack_list)
    return jsonify({'status': 'success', 'message': 'Attack list received!', 'received_data': attack_list})


@app.route('/api/execute_attack', methods=['POST'])
@auth
def execute_attack():
    try:
        data = request.json
        username = data.get('username', None).strip('"')
        print(f"Executing attack for user: {username}")

        # 下游任务执行代码
        initTime = int(time.time())
        date = str(datetime.now())[:10]
        attack = AttackRecord(createUserName=username, createTime=initTime, attackResult=json.dumps("RUNNING"))
        initTime = str(initTime)
        db.session.add(attack)
        db.session.commit()
        project_path = os.path.dirname(os.path.abspath(__file__))
        try:
            model_class = parse_config(project_path, initTime, str(username))
            model_class.run()

            fileName = username + '_single_' + initTime + '_' + date + '.txt'
            result = extractResult(os.path.join(project_path, 'logs', fileName))
            AttackRecord.query.filter_by(attackID=attack.attackID, createUserName=username).update(
                {AttackRecord.attackResult: json.dumps(result)})
            db.session.commit()
            return jsonify({'status': 'success', 'message': 'Attack executed successfully!'})
        except Exception as executeError:
            AttackRecord.query.filter_by(attackID=attack.attackID, createUserName=username).update(
                {AttackRecord.attackResult: json.dumps("FAILED")})
            raise executeError

    except Exception as e:
        print("Error executing attack:", e)
        return jsonify({'status': 'error', 'message': 'Failed to execute attack'}), 500


@app.route('/api/getRecord', methods=['POST'])
@auth
def getRecord():
    try:
        data = request.json
        username = data.get('username', None).strip('"')
        createTime = data.get('createTime', None)
        # attackID = data.get('attackID', None)

        attackRecord = AttackRecord.query.filter_by(createUserName=username, createTime=createTime).first()
        # if attackID is None:
        #     attackRecord = AttackRecord.query.filter_by(createUserName=username).order_by(AttackRecord.createTime.desc()).first()
        # else:
        #     attackRecord = AttackRecord.query.filter_by(attackID=attackID, createUserName=username).first()
        # result = attackRecord.attackResult
        result = json.loads(attackRecord.attackResult)
        return jsonify(result), 200
    except Exception as e:
        print(f"Error fetching log: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/attackRecords', methods=['POST'])
@auth
def attackRecords():
    data = request.json
    username = data.get('username', None).strip('"')
    currentPageSize = data.get('currentPageSize') #可调
    currentPage = data.get('currentPage', None)
    onlyTreasure = data.get('onlyTreasure', False)

    try:
        # direct = "./logs"
        # fileList = []
        # timeStampList = []
        # for fileName in os.listdir(direct):
        #     match = re.match(username + r'_([A-Za-z]*_)?(\d+)_\d{4}-\d{2}-\d{2}', fileName);
        #     if match:
        #         fileList.append(fileName)
        #         timeStampList.append(eval(match.group(2)))
        # totalRecordsNum = len(fileList)
        # if totalRecordsNum == 0:
        #     print("不存在合法记录")
        #     return jsonify({'records': [], 'pagination': {'totalRecordsNum': totalRecordsNum}}), 200
        # records = sorted(zip(timeStampList, fileList), reverse=True)[(currentPage - 1) * currentPageSize:currentPage * currentPageSize]
        totalRecordsNum = AttackRecord.query.filter_by(createUserName=username).count()
        if totalRecordsNum != 0:
            if onlyTreasure:
                records = (AttackRecord.query.filter_by(createUserName=username, isTreasure=True).
                           order_by(AttackRecord.createTime.desc()).offset((currentPage - 1) * currentPageSize).limit(
                    currentPageSize).all())
            else:
                records = (AttackRecord.query.filter_by(createUserName=username).
                           order_by(AttackRecord.createTime.desc()).offset((currentPage - 1) * currentPageSize).limit(
                    currentPageSize).all())
            records = list(map(
                lambda r: (r.createTime, 0 if r.attackResult == json.dumps("RUNNING") else (
                    2 if r.attackResult == json.dumps("FAILED") else 1), r.isTreasure),
                records))
        else:
            records = []
        retData = {'records': records, 'pagination': {'totalRecordsNum': totalRecordsNum}}
        return jsonify(retData), 200
    except Exception as e:
        print(f"Error fetching log: {e}")
        return jsonify({"status": 'error', "message": "Failed to fetch records"}), 500


'''删除一条记录，同时删除日志文件与数据库记录。不存在日志文件时会报错。'''
@app.route('/api/deleteRecord', methods=['POST'])
@auth
def deleteRecord():
    try:
        data = request.json
        username = data.get('username', None).strip('"')
        createTime = data.get('createTime', None)
        # attackID = data.get('attackID', None)

        count = AttackRecord.query.filter_by(createUserName=username, createTime=createTime).delete()
        db.session.commit()
        fileName = username + '_single_' + str(createTime) + '_' + str(datetime.now())[:10] + '.txt'
        filePath = os.path.join(project_path, 'logs', fileName)
        os.remove(filePath)
        if (count == 1):
            return jsonify({'status': 'success', 'message': 'Delete executed successfully!'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'failed to delete record'}), 500
    except Exception as e:
        print(f"Error fetching log: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# 收藏
@app.route('/api/treasure', methods=['POST'])
@auth
def treasure():
    try:
        data = request.json
        username = data.get('username', None).strip('"')
        createTime = data.get('createTime', None)
        isTreasure = data.get('isTreasure', False)
        count = AttackRecord.query.filter_by(createUserName=username, createTime=createTime).update({AttackRecord.isTreasure: isTreasure})
        db.session.commit()
        if count == 0:
            return jsonify({'status': 'error', 'message': 'failed to treasure'}), 500
        else:
            return jsonify({'status': 'success', 'message': 'treasure executed successfully!'}), 200
    except Exception as e:
        print(f"Error fetching log: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(port=46666, debug=True)
