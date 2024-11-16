import base64
import hmac
import hashlib
import json
import time


# Base64Url 编码函数
def base64UrlEncode(source):
    # 将字节数据进行 Base64 编码，然后去掉 '='，替换 '+' 为 '-'，替换 '/' 为 '_'
    base64_str = base64.b64encode(source).decode('utf-8')
    base64_url = base64_str.rstrip('=')  # 去掉尾部的 '='
    base64_url = base64_url.replace('+', '-')  # 将 '+' 替换为 '-'
    base64_url = base64_url.replace('/', '_')  # 将 '/' 替换为 '_'
    return base64_url


# 签名函数
def sign(payload, salt, expiresIn):
    # 获取当前时间的 UNIX 时间戳
    exp = int(time.time()) + expiresIn

    # 构建 JWT 的 header 和 payload
    header = {"alg": "HS256", "type": "JWT", "exp": exp}
    payload["exp"] = exp  # 将过期时间添加到 payload 中

    # 将 header 和 payload 转换为 JSON 字符串
    header_str = json.dumps(header)
    payload_str = json.dumps(payload)

    # 对 header 和 payload 进行 Base64Url 编码
    base64_header = base64UrlEncode(header_str.encode('utf-8'))
    base64_payload = base64UrlEncode(payload_str.encode('utf-8'))

    # 创建 Base64Url 编码后的 token 部分
    base64_str = base64_header + '.' + base64_payload

    # 使用 HMAC 生成签名
    signature = hmac.new(salt.encode('utf-8'), base64_str.encode('utf-8'), hashlib.sha256).digest()
    base64_signature = base64UrlEncode(signature)

    # 返回完整的 JWT
    return base64_str + '.' + base64_signature

