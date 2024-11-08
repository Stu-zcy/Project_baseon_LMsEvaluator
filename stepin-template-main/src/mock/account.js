import Mock from 'mockjs';
import CryptoJS from 'crypto-js';
import axios from 'axios';

function base64UrlEncode(source) {
  return CryptoJS.enc.Base64.stringify(source).replace(/=+$/, '').replace(/\+/g, '-').replace(/\//g, '_');
}

function sign(payload, salt, { expiresIn }) {
  const exp = Math.floor(Date.now() / 1000) + expiresIn;
  const header = JSON.stringify({ alg: 'HS256', type: 'JWT', exp });
  const payloadStr = JSON.stringify({ ...payload, exp });

  const base64Str =
    base64UrlEncode(CryptoJS.enc.Utf8.parse(header)) + '.' + base64UrlEncode(CryptoJS.enc.Utf8.parse(payloadStr));
  const signature = base64UrlEncode(CryptoJS.HmacSHA256(base64Str, salt));
  return base64Str + '.' + signature;
}

let globalAccountInfo = {
  username: 'iczer',
  role: '',
  age: 0,
  gender: 0,
  permissions: []
};

// 这个部分只用于 Mock 数据
Mock.mock('api/login', 'post', ({ body }) => {
    const { username, password } = JSON.parse(body ?? '{}');
  
    // 这里模拟成功返回
    const expiresIn = 24 * 60 * 60 * 1000;
    const accountInfo = {
      role: 'admin',
      age: 30,
      gender: 1,
      permissions: ['edit', 'delete', 'add']
    };
  
    return {
      code: 0,
      message: 'success',
      data: { accountInfo, expiresIn }
    };
  });
  
  // 实际应用中的请求逻辑
function login(username, password) {
    fetch('http://localhost:5000/api/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    })
    .then(response => response.json())
    .then(data => {
      if (data.code === 0) {
        const { message, accountInfo, expiresIn } = data.data;
        globalAccountInfo = { ...accountInfo, username };
        const token = sign({ username, role: accountInfo.role }, 'secret key', { expiresIn });
        console.log('Generated token:', token);
        // 进行额外操作，例如导航到主界面
      } else {
        console.log('Error:', data.message);
      }
    })
    .catch(error => {
      console.error('Error occurred:', error);
    });
  }
  

  
  
  

Mock.mock('api/account', 'get', () => {
  return {
    code: 0,
    message: 'success',
    data: {
      account: {
        username: globalAccountInfo.username,
        age: globalAccountInfo.age || 18,
        gender: globalAccountInfo.gender || 0,
        avatar: 'http://portrait.gitee.com/uploads/avatars/user/691/2073535_iczer_1578965604.png!avatar30',
      },
      role: globalAccountInfo.role || 'admin',
      permissions: globalAccountInfo.permissions || ['edit', 'delete', 'add'],
    },
  };
});
