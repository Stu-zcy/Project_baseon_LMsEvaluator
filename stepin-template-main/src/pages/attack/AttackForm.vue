<script setup>
import { reactive, ref } from 'vue';
import axios from 'axios';

// 定义所有攻击参数的表单数据结构
const attackArgs = reactive({
  attack: false,
  attack_type: 'AdvAttack',
  attack_recipe: 'BAEGarg2019',
  use_local_model: false,
  use_local_tokenizer: false,
  use_local_dataset: false,
  model_name_or_path: 'LMs/bert_base_uncased_english',
  tokenizer_name_or_path: 'LMs/bert_base_uncased_english',
  dataset_name_or_path: 'data/imdb/test.txt',
  attack_nums: 2,
  display_full_info: false
});

const showForm = ref(false);
const loading = ref(false);

function submit() {
  loading.value = true;

  // 将攻击参数组合成 attack_list
  const attackList = Object.entries(attackArgs).map(([key, value]) => ({
    name: key,
    value: value
  }));

  axios.post('http://localhost:5000/api/data', { attack_list: attackList })
    .then(response => {
      console.log("Response:", response.data);
      showForm.value = false;
    })
    .finally(() => {
      loading.value = false;
    });
}
</script>

<template>
  <div class="authority">
    <a-button type="primary" @click="() => showForm.value = true">设置攻击参数</a-button>

    <a-modal :okButtonProps="{ loading }" width="540px" v-model:visible="showForm" title="攻击配置" @ok="submit">
      <a-form :model="attackArgs" label-col="{ span: 6 }" wrapper-col="{ span: 16 }">
        <!-- 基本设置 -->
        <a-form-item label="开启攻击">
          <a-switch v-model:checked="attackArgs.attack" checked-children="是" un-checked-children="否" />
        </a-form-item>
        <a-form-item label="使用本地模型">
          <a-switch v-model:checked="attackArgs.use_local_model" checked-children="是" un-checked-children="否" />
        </a-form-item>
        <a-form-item label="显示全部过程信息">
          <a-switch v-model:checked="attackArgs.display_full_info" checked-children="是" un-checked-children="否" />
        </a-form-item>

        <!-- 高级设置 -->
        <a-collapse>
          <a-collapse-panel header="高级设置" key="1">
            <a-form-item label="攻击类型">
              <a-input v-model:value="attackArgs.attack_type" placeholder="输入攻击类型" />
            </a-form-item>
            <a-form-item label="攻击策略">
              <a-input v-model:value="attackArgs.attack_recipe" placeholder="输入攻击策略" />
            </a-form-item>
            <a-form-item label="模型路径或名称">
              <a-input v-model:value="attackArgs.model_name_or_path" placeholder="模型路径或 Huggingface 名称" />
            </a-form-item>
            <a-form-item label="tokenizer路径或名称">
              <a-input v-model:value="attackArgs.tokenizer_name_or_path" placeholder="Tokenizer 路径或 Huggingface 名称" />
            </a-form-item>
            <a-form-item label="数据集路径或名称">
              <a-input v-model:value="attackArgs.dataset_name_or_path" placeholder="数据集路径或 Huggingface 名称" />
            </a-form-item>
            <a-form-item label="攻击次数">
              <a-input-number v-model:value="attackArgs.attack_nums" min="1" placeholder="输入攻击次数" />
            </a-form-item>
          </a-collapse-panel>
        </a-collapse>
      </a-form>
    </a-modal>
  </div>
</template>

<style scoped>
.authority {
  padding: 20px;
}
</style>
