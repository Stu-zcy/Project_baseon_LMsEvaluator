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
        <a-form-item :required="true" name="email">
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
          />
        </a-form-item>

        <!-- 性别输入框 -->
        <a-form-item label="性别" name="gender">
          <a-radio-group v-model:value="form.gender" class="signin-radio-group">
            <a-radio value="0">男</a-radio>
            <a-radio value="1">女</a-radio>
            <a-radio value="2">其他</a-radio>
          </a-radio-group>
        </a-form-item>

        <!-- 年龄输入框 -->
        <a-form-item label="年龄" name="age">
          <a-input-number v-model:value="form.age" class="signin-input-number" min="0" placeholder="请输入年龄" />
        </a-form-item>

        <!-- 验证码输入框 -->
        <a-form-item label="验证码" name="verificationCode">
          <div class="verification-container">
            <a-input
              v-model:value="form.verificationCode"
              placeholder="请输入验证码"
              class="signin-input h-[40px]"
            />
            <a-button @click="requestMailcode" type="primary" class="ml-2">
              发送验证码
            </a-button>
          </div>
        </a-form-item>

        <!-- 注册按钮 -->
        <a-button htmlType="submit" class="submit-button" type="primary" :loading="loading"> 注册 </a-button>
      </a-form>
    </div>
  </ThemeProvider>
</template>

<script lang="ts" setup>
import { reactive, ref } from 'vue';
import { message } from 'ant-design-vue';
import axios from 'axios';

const loading = ref(false);
const form = reactive({
  username: '',
  email: '',
  password: '',
  age: null,
  gender: undefined, // 确保初始化为 undefined
  verificationCode: '',
});

async function requestMailcode() {
  try {
    const response = await axios.post('http://localhost:5000/api/send_verification_code', { email: form.email });
    message.success('验证码已发送，请查收邮箱！');
  } catch (error) {
    throw new Error(error.response?.data?.message || '发送验证码失败');
  }
}

async function requestSignin() {
  loading.value = true;
  try {
    const response = await axios.post('http://localhost:5000/api/register', form);
    message.success('注册成功！');
  } catch (error) {
    throw new Error(error.response?.data?.message || '注册失败');
  } finally {
    loading.value = false;
  }
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

.signin-input,
.signin-select,
.signin-radio-group,
.signin-input-number {
  margin-bottom: 16px;
  border-radius: 8px;
  padding-left: 12px;
  font-size: 14px;
  height: 40px;
}

.signin-radio-group {
  display: flex;
  gap: 20px;
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

/* 使性别和年龄输入框分为两行 */
a-form-item {
  width: 100%;
}

@media (max-width: 768px) {
  .signin-box {
    width: 100%;
    padding: 20px;
  }

  .verification-container {
    flex-direction: column;
    gap: 10px;
  }
}
</style>
