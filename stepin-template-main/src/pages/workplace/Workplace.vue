<template>
  <div id="app">
    <h1>攻击参数配置</h1>
    <AttackForm />
  </div>
</template>

<script>
import AttackForm from './AttackForm.vue';

export default {
  components: {
    AttackForm
  }
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
