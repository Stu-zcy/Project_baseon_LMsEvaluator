<script lang="ts" setup>
  import { GithubFilled } from '@ant-design/icons-vue';
  import { useRouter } from 'vue-router';

  const router = useRouter();
  function signIn() {
    router.push('/login');
  }
</script>
<template>
  <div class="home text-center flex flex-col justify-center">
    <div class="transform w-full">
      <div class="tracking-wide slogan text-[3.5rem] xl:text-[5.25rem] font-extralight">
        <span class="font-semibold">LMsEvaluator</span>
      </div>
      <p class="text-subtext text-[1.5rem] xl:text-[1.75rem] font-extralight tracking-wide">
        本项目是一个基于PyTorch的大模型评估平台，包含了以BERT为主的各种大模型，以及多个下游任务在安全与隐私风险两方面的评估
      </p>
      <button
        @click="signIn"
        class="bg-primary-300 hover:bg-primary-400 cursor-pointer mt-lg shadow border-0 outline-none text-xl px-[64px] py-lg rounded-sm"
      >
        Sign In and Start
        <br />
      </button>
      <br />
      <div class="inline-block text-gray-200 text-xl mr-lg">Powered by:</div>
      <div class="powered-by-list inline-flex mt-xl text-subtext">
        <a 
          href="https://github.com/Kyaruk/LMsEvaluator/" target="_blank" class="powered-by text-subtext hover:text-text cursor-pointer">
          <GithubFilled /> LMsEvaluator  
        </a>
        <a 
          href="https://www.antdv.com/" target="_blank" class="powered-by text-subtext hover:text-text cursor-pointer">
          <img class="w-6" src="@/assets/logo/antdv.svg" />
          antd design vue
        </a>
        
      </div>
    </div>
  </div>
</template>
<style scoped lang="less">
  .home {
    min-height: max(100vh, 600px);
    margin-top: -78px;
  }
</style>
