<template>
  <ThemeProvider :color="{ middle: { 'bg-base': '#fff' }, primary: { DEFAULT: '#1896ff' } }">
    <div class="signin-box rounded-sm">
      <a-form
        :model="form"
        :wrapperCol="{ span: 24 }"
        @finish="requestSignin"
        class="signin-form w-[400px] p-lg xl:w-[440px] xl:p-xl h-fit text-text"
      >
        <a-divider class="form-divider">账户注册</a-divider>

        <!-- 用户名输入框 -->
        <a-form-item :required="true" name="username">
          <a-input
            v-model:value="form.username"
            placeholder="请输入用户名"
            class="signin-input h-[40px]"
          />
        </a-form-item>

        <!-- 邮箱输入框 -->
        <a-form-item :required="true" name="email" :rules="[{ type: 'email', message: '请输入正确的邮箱格式' }]">
          <a-input
            v-model:value="form.email"
            placeholder="请输入邮箱"
            class="signin-input h-[40px]"
          />
        </a-form-item>

        <!-- 密码输入框 -->
        <a-form-item :required="true" name="password">
          <a-input
            v-model:value="form.password"
            placeholder="请输入密码"
            class="signin-input h-[40px]"
            type="password"
            @input="checkPasswordStrength"
          />
        </a-form-item>

        <!-- 确认密码输入框 -->
        <a-form-item :required="true" name="confirmPassword" :rules="[{ validator: validateConfirmPassword, message: '两次密码输入不一致' }]">
          <a-input
            v-model:value="form.confirmPassword"
            placeholder="请再次输入密码"
            class="signin-input h-[40px]"
            type="password"
          />
          <div class="password-strength-bar">
            <div :class="['strength-bar', passwordStrengthClass]"></div>
          </div>
          <div class="password-strength-text">{{ passwordStrengthText }}</div>
        </a-form-item>

        <!-- 验证码输入框 -->
        <a-form-item label="验证码" name="verificationCode">
          <div class="verification-container">
            <a-input
              v-model:value="form.verificationCode"
              placeholder="请输入验证码"
              class="signin-input h-[40px]"
            />
            <a-button @click="requestMailcode" type="primary" class="ml-2 mb-4">
              发送验证码
            </a-button>
          </div>
        </a-form-item>

        <!-- 头像选择 -->
        <a-form-item label="选择头像" name="avatar">
          <div class="avatar-container">
            <div
              v-for="(avatar, index) in avatarList"
              :key="index"
              class="avatar-item"
              :class="{ selected: form.avatar === avatar }"
              @click="selectAvatar(avatar)"
            >
              <img :src="avatar" alt="Avatar" class="avatar-img" />
            </div>
          </div>
        </a-form-item>

        <!-- 注册按钮 -->
        <a-button htmlType="submit" class="submit-button" type="primary" :loading="loading"> 注册 </a-button>
      </a-form>
      <!-- 返回按钮 -->
      <a-button @click="goBack" class="return-button" type="default"> 返回 </a-button>
    </div>
  </ThemeProvider>
</template>

<script lang="ts" setup>
import { reactive, ref, defineEmits } from 'vue';
import { message } from 'ant-design-vue';
import axios from 'axios';
import { useRouter } from 'vue-router';
const router = useRouter();
const emit = defineEmits(['success', 'goBack']);
const loading = ref(false);
const form = reactive({
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
  verificationCode: '',
  avatar: '', // 新增字段用于存储头像
});

const passwordStrengthText = ref('');
const passwordStrengthClass = ref('');

// 头像列表
const avatarList = Array.from({ length: 20 }, (_, index) => `https://gitee.com/topiza/image/raw/master/file_${index + 1}.png`);

async function requestMailcode() {
  try {
    const response = await axios.post('http://127.0.0.1:5000/api/send_verification_code', { email: form.email });
    message.success('验证码已发送，请查收邮箱！');
  } catch (error) {
    throw new Error(error.response?.data?.message || '发送验证码失败');
  }
}

async function requestSignin() {
  loading.value = true;
  try {
    // 向后端发送请求
    const response = await axios.post('http://127.0.0.1:5000/api/register', form);

    // 根据后端返回的消息处理
    if (response.status === 201) {
      message.success('注册成功！');
      emit('success');
      goBack(); // 注册成功后返回登录页面
    }
  } catch (error) {
    // 根据不同的错误信息展示不同的提示
    if (error.response) {
      // 错误响应包含后端返回的消息
      const messageContent = error.response?.data?.message || '注册失败';
      message.error(messageContent);
    } else {
      // 没有响应，可能是网络问题
      message.error('请求失败，请检查网络连接');
    }
  } finally {
    loading.value = false;
  }
}


function checkPasswordStrength() {
  const lengthScore = form.password.length >= 8 ? 1 : 0;
  const hasNumbers = /\d/.test(form.password) ? 1 : 0;
  const hasUppercase = /[A-Z]/.test(form.password) ? 1 : 0;
  const hasSpecialChars = /[!@#$%^&*]/.test(form.password) ? 1 : 0;
  const score = lengthScore + hasNumbers + hasUppercase + hasSpecialChars;

  if (score === 4) {
    passwordStrengthText.value = '强';
    passwordStrengthClass.value = 'strength-strong';
  } else if (score >= 2) {
    passwordStrengthText.value = '一般';
    passwordStrengthClass.value = 'strength-medium';
  } else {
    passwordStrengthText.value = '弱';
    passwordStrengthClass.value = 'strength-weak';
  }
}

function validateConfirmPassword(_, value) {
  if (value !== form.password) {
    return Promise.reject('两次密码输入不一致');
  }
  return Promise.resolve();
}

function selectAvatar(avatar: string) {
  form.avatar = avatar;
}

function goBack() {
  router.push('/home');
}
</script>

<style scoped>
.signin-box {
  padding: 30px;
  border-radius: 10px;
  background-color: #ffffff;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  max-width: 500px;
  margin: auto;
}

.form-divider {
  margin-bottom: 20px;
  font-size: 18px;
  font-weight: bold;
  color: #333;
}

.signin-input {
  margin-bottom: 16px;
  border-radius: 8px;
  padding-left: 12px;
  font-size: 14px;
  height: 40px;
}

.verification-container {
  display: flex;
  align-items: center;
}

.submit-button {
  width: 100%;
  height: 45px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  margin-top: 20px;
}

.return-button {
  width: 100%;
  height: 45px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  margin-top: 10px;
}

.password-strength-bar {
  height: 6px;
  border-radius: 5px;
  margin-top: 8px;
  background-color: #e0e0e0;
}

.strength-bar {
  height: 100%;
  border-radius: 5px;
  transition: width 0.3s ease;
}

.strength-weak {
  width: 33%;
  background-color: #ff4d4f;
}

.strength-medium {
  width: 66%;
  background-color: #faad14;
}

.strength-strong {
  width: 100%;
  background-color: #52c41a;
}

.password-strength-text {
  font-size: 14px;
  margin-top: 5px;
  color: #1896ff;
}

.avatar-container {
  display: flex;
  overflow-x: auto;
  margin-top: 16px;
}

.avatar-item {
  margin-right: 10px;
  cursor: pointer;
  transition: transform 0.3s;
}

.avatar-item:hover {
  transform: scale(1.1);
}

.avatar-item.selected {
  border: 2px solid #1896ff;
  padding: 2px;
}

.avatar-img {
  width: 40px;
  height: 40px;
  border-radius: 50%;
}

.verification-container .signin-input {
  width: 200px;
}
</style>
