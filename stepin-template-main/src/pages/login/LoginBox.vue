<template>
  <ThemeProvider :color="{ middle: { 'bg-base': '#fff' }, primary: { DEFAULT: '#1896ff' } }">
    <div class="login-box rounded-lg shadow-lg p-[20px] w-[400px] mx-auto my-10 bg-white">
      <a-form
        v-if="!isSigningIn"
        :model="form"
        :wrapperCol="{ span: 24 }"
        @finish="login"
        class="login-form"
      >
        <a-divider class="text-center">账户登录</a-divider>
        
        <!-- 用户名 -->
        <a-form-item :required="true" name="username" class="mb-6">
          <a-input
            v-model:value="form.username"
            autocomplete="new-username"
            placeholder="请输入用户名或邮箱"
            class="login-input h-[40px] border border-gray-300 rounded-lg shadow-sm"
          />
        </a-form-item>
        
        <!-- 密码 -->
        <a-form-item :required="true" name="password" class="mb-6">
          <a-input
            v-model:value="form.password"
            autocomplete="new-password"
            placeholder="请输入登录密码"
            class="login-input h-[40px] border border-gray-300 rounded-lg shadow-sm"
            type="password"
          />
        </a-form-item>

        <!-- 登录按钮 -->
        <a-button 
          htmlType="submit" 
          class="h-[40px] w-full text-white bg-blue-500 hover:bg-blue-600 rounded-lg shadow-md transition duration-300" 
          :loading="loading">
          登录
        </a-button>

        <a-divider class="my-6"></a-divider>

        <!-- 条款说明 -->
        <div class="terms text-center text-sm text-gray-600">
          登录即代表您同意我们的
          <span class="font-bold">用户条款</span>、<span class="font-bold">数据使用协议</span>、以及
          <span class="font-bold">Cookie使用协议</span>。
        </div>

        <!-- 注册按钮 -->
        <a-button 
          class="register-button w-full py-2 mt-4 text-blue-500 hover:text-blue-700 border-none bg-transparent"
          @click="toggleSignIn">
          注册新账户
        </a-button>
      </a-form>

      <!-- 注册表单 -->
      <SigninBox v-else @success="onRegisterSuccess" @failure="onRegisterFailure" />
    </div>
  </ThemeProvider>
</template>

<script lang="ts" setup>
import { reactive, ref } from 'vue';
import { useAccountStore } from '@/store';
import { ThemeProvider } from 'stepin';
import { message } from 'ant-design-vue';
import SigninBox from './SigninBox.vue';

const loading = ref(false);
const isSigningIn = ref(false); // 新增状态管理

const form = reactive({
  username: undefined,
  password: undefined,
});

const emit = defineEmits<{
  (e: 'success', fields: any): void;
  (e: 'failure', reason: string, fields: any): void;
}>();

const accountStore = useAccountStore();

function login(params: { username: string; password: string }) {
  loading.value = true;
  accountStore
    .login(params.username, params.password)
    .then((res) => {
      emit('success', params);
      message.success('登录成功！');
    })
    .catch((error) => {
      emit('failure', error.message || '登录失败', params);
      message.error(error.message || '登录失败');
    })
    .finally(() => {
      loading.value = false;
    });
}

function toggleSignIn() {
  isSigningIn.value = !isSigningIn.value; // 切换注册和登录状态
}

function onRegisterSuccess() {
  message.success('注册成功！');
  toggleSignIn(); // 注册成功后切换回登录页面
}

function onRegisterFailure(reason: string) {
  message.error(reason);
}
</script>

<style scoped lang="less">
/* 调整注册按钮的样式 */
.register-button {
  margin-top: 10px; /* 添加适当的间距 */
  font-size: 14px;
  font-weight: bold;
}

/* 更改登录框的样式 */
.login-box {
  max-width: 520px; /* 最大宽度 */
  width: 100%; /* 使登录框的宽度响应式 */
  background-color: #fff; /* 白色背景 */
  border-radius: 10px; /* 圆角 */
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); /* 阴影 */
  padding: 20px;
}

/* 调整表单项的间距 */
a-form-item {
  margin-bottom: 16px;
}

/* 调整输入框的样式 */
.login-input {
  border-radius: 8px;
  padding-left: 15px;
}

/* 调整按钮的样式 */
a-button {
  border-radius: 8px;
  font-size: 16px;
  height: 40px;
}

/* 条款文本的样式 */
.terms {
  font-size: 14px;
  color: #555;
}

/* 登录框内文字的样式 */
.text-center {
  text-align: center;
}
</style>
