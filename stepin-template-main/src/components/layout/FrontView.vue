<script lang="ts" setup>
  import { LogoutOutlined } from '@ant-design/icons-vue';
  import { onMounted } from 'vue';
  import { ThemeProvider, alert } from 'stepin';



  const navList = [
    {
      title: 'Products',
      children: [
        {
          title: 'Stepin Template',
          list: ['Stepin Pro', 'Stepin Style', 'Stepin Admin'],
        },
        {
          title: 'Stepin',
          list: ['Stepin Vue', 'Stepin React', 'Stepin Angular'],
        },
      ],
    }
  ];
</script>
<template>
  <ThemeProvider :color="{ middle: { 'bg-base': '#003f8c' }, primary: { DEFAULT: '#1896ff' } }" :autoAdapt="false">
    <div class="front-view flex flex-col">
      <div class="text-text flex-1">
        <div class="front-header flex items-baseline py-md px-xl">
          <router-link to="/home" class="text-xxl text-text hover:text-text">
            <img src="@/assets/vite.svg" />
            LMsEvaluator
          </router-link>
          <div
            style="width: calc(100% - 430px)"
            class="front-navigation mx-xl flex overflow-hidden items-center text-lg overflow-ellipsis whitespace-nowrap"
          >
          </div>
        </div>
        <div class="front-content px-xl">
          <router-view />
        </div>
      </div>
    </div>
  </ThemeProvider>
</template>
<style lang="less" scoped>
  .front-view {
    .front-header {
      .front-nav-item {
        &.with-list .front-nav-item-content {
          &:after {
            content: '';
            @apply ~"h-[8px]" ~"w-[8px]" transition-transform ml-2 inline-block border-text border-l-0 border-t-0 border-r-2 border-b-2 border-solid ~"rotate-[-135deg]" translate-y-1/4;
          }
          &:hover {
            &:after {
              @apply ~"rotate-[45deg]" translate-y-0;
            }
          }
        }
      }
    }
    .front-content {
      min-height: calc(100vh - 78px);
    }
  }
</style>
