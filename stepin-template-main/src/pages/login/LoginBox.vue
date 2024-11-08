<template>
  <ThemeProvider :color="{ middle: { 'bg-base': '#fff' }, primary: { DEFAULT: '#1896ff' } }">
    <div class="login-box rounded-sm">
      <a-form
        v-if="!isSigningIn"
        :model="form"
        :wrapperCol="{ span: 24 }"
        @finish="login"
        class="login-form w-[400px] p-lg xl:w-[440px] xl:p-xl h-fit text-text"
      >
        <a-divider>账户登录</a-divider>
        <a-form-item :required="true" name="username">
          <a-input
            v-model:value="form.username"
            autocomplete="new-username"
            placeholder="请输入用户名或邮箱"
            class="login-input h-[40px]"
          />
        </a-form-item>
        <a-form-item :required="true" name="password">
          <a-input
            v-model:value="form.password"
            autocomplete="new-password"
            placeholder="请输入登录密码"
            class="login-input h-[40px]"
            type="password"
          />
        </a-form-item>
        <a-button htmlType="submit" class="h-[40px] w-full" type="primary" :loading="loading"> 登录 </a-button>
        
        <a-divider></a-divider>
        <div class="terms">
          登录即代表您同意我们的
          <span class="font-bold">用户条款 </span>、<span class="font-bold"> 数据使用协议 </span>、以及
          <span class="font-bold">Cookie使用协议</span>。
        </div>
        <a-button class="register-button" @click="toggleSignIn">注册新账户</a-button>
      </a-form>
      
      <SigninBox v-else @success="onRegisterSuccess" @failure="onRegisterFailure" />
    </div>
  </ThemeProvider>
</template>

<script lang="ts" setup>
import { reactive, ref } from 'vue';
import { useAccountStore } from '@/store';
import { ThemeProvider } from 'stepin';
import { message } from 'ant-design-vue';
import SigninBox from './SigninBox.vue'; // 导入 SigninBox

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
.register-button {
  margin-top: 10px; /* 添加适当的间距 */
}
</style>
