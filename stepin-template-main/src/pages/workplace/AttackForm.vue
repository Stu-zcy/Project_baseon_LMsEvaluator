<template>
  <div class="attack-configuration">
    <div class="header">
      <a-button type="primary" icon="plus" @click="showAddModal = true">添加表单</a-button>
    </div>

    <a-table :columns="columns" :dataSource="attackList" rowKey="index" class="attack-table">
      <template #bodyCell="{ column, record, index }">
        <span v-if="column.dataIndex === 'operation'">
          <a-button type="link" @click="editAttackArgs(index)">编辑</a-button>
          <a-button type="link" danger @click="removeAttackArgs(index)">删除</a-button>
        </span>
      </template>
    </a-table>

    <a-modal v-model:visible="showAddModal" title="添加攻击配置" @ok="addAttackArgs" @cancel="cancelAddModal" width="600px">
      <div class="input-section">
        <label class="input-label">
          攻击:
          <a-switch v-model:checked="currentAttackArgs.attack" />
        </label>
        <label class="input-label">
          攻击类型:
          <a-select v-model="currentAttackArgs.attack_type" @change="updateAttackRecipe" style="width: 100%;">
            <a-select-option v-for="type in attackTypes" :key="type" :value="type">{{ type }}</a-select-option>
          </a-select>
        </label>
        <label class="input-label">
          攻击策略:
          <a-select v-model="currentAttackArgs.attack_recipe" style="width: 100%;">
            <a-select-option v-for="recipe in attackRecipes" :key="recipe" :value="recipe">{{ recipe }}</a-select-option>
          </a-select>
        </label>

        <!-- 高级设置 -->
        <a-collapse>
          <a-collapse-panel header="高级设置" key="1">
            <div class="advanced-settings">
              <div class="setting-item">
                <label class="input-label">使用本地模型:</label>
                <a-switch v-model:checked="currentAttackArgs.use_local_model" />
              </div>
              <div class="setting-item">
                <label class="input-label">使用本地Tokenizer:</label>
                <a-switch v-model:checked="currentAttackArgs.use_local_tokenizer" />
              </div>
              <div class="setting-item">
                <label class="input-label">使用本地数据集:</label>
                <a-switch v-model:checked="currentAttackArgs.use_local_dataset" />
              </div>

              <div class="setting-item">
                <label class="input-label">模型名称或路径:</label>
                <input type="file" webkitdirectory @change="(event) => handleFolderUpload(event, 'model')" class="file-input" />
                <span v-if="currentAttackArgs.model_name_or_path" class="upload-success">上传成功: {{ currentAttackArgs.model_name_or_path }}</span>
              </div>

              <div class="setting-item">
                <label class="input-label">Tokenizer 名称或路径:</label>
                <input type="file" webkitdirectory @change="(event) => handleFolderUpload(event, 'tokenizer')" class="file-input" />
                <span v-if="currentAttackArgs.tokenizer_name_or_path" class="upload-success">上传成功: {{ currentAttackArgs.tokenizer_name_or_path }}</span>
              </div>

              <div class="setting-item">
                <label class="input-label">数据集名称或路径:</label>
                <input type="file" webkitdirectory @change="(event) => handleFolderUpload(event, 'dataset')" class="file-input" />
                <span v-if="currentAttackArgs.dataset_name_or_path" class="upload-success">上传成功: {{ currentAttackArgs.dataset_name_or_path }}</span>
              </div>

              <div class="setting-item">
                <label class="input-label">攻击次数:</label>
                <input type="number" v-model.number="currentAttackArgs.attack_nums" placeholder="攻击次数" class="rounded-input" />
              </div>
              <div class="setting-item">
                <label class="input-label">显示完整信息:</label>
                <a-switch v-model:checked="currentAttackArgs.display_full_info" />
              </div>
            </div>
          </a-collapse-panel>
        </a-collapse>
      </div>
    </a-modal>

    <div class="action-buttons">
      <a-button type="primary" @click="sendAttackList">发送所有配置</a-button>
      <a-button type="default" @click="executeAttack">执行攻击</a-button>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive } from 'vue';
import axios from 'axios';
import { message } from 'ant-design-vue';

// 默认攻击配置模板
const defaultAttackArgs = () => ({
  index: 0,
  attack: false,
  attack_type: 'AdvAttack',
  attack_recipe: 'BAEGarg2019',
  use_local_model: true,
  use_local_tokenizer: true,
  use_local_dataset: true,
  model_name_or_path: '',
  tokenizer_name_or_path: '',
  dataset_name_or_path: '',
  attack_nums: 2,
  display_full_info: true,
});

// 攻击类型及对应的攻击策略
const attackOptions = {
  GIAforNLP: ['default'],
  SWAT: ['default'],
  AdvAttack: [
    'A2TYoo2021', 'BAEGarg2019', 'BERTAttackLi2020', 'GeneticAlgorithmAlzantot2018',
    'FasterGeneticAlgorithmJia2019', 'DeepWordBugGao2018', 'HotFlipEbrahimi2017',
    'InputReductionFeng2018', 'Kuleshov2017', 'MorpheusTan2020',
    'Seq2SickCheng2018BlackBox', 'TextBuggerLi2018', 'TextFoolerJin2019',
    'PWWSRen2019', 'IGAWang2019', 'Pruthi2019', 'PSOZang2020',
    'CheckList2020', 'CLARE2020', 'FrenchRecipe', 'SpanishRecipe', 'ChineseRecipe'
  ],
  BackDoorAttack: ['default'],
  PoisoningAttack: ['default']
};

// 攻击配置列表
const attackList = ref([]);

// 当前输入的攻击配置
const currentAttackArgs = reactive(defaultAttackArgs());

// 控制添加表单的提示框
const showAddModal = ref(false);

// 表格列配置
const columns = [
  { title: '序号', dataIndex: 'index' },
  { title: 'Attack Type', dataIndex: 'attack_type' },
  { title: 'Attack Recipe', dataIndex: 'attack_recipe' },
  { title: 'Model Name or Path', dataIndex: 'model_name_or_path' },
  { title: '操作', dataIndex: 'operation' },
];

// 攻击类型选项
const attackTypes = Object.keys(attackOptions);

// 当前攻击策略选项
const attackRecipes = ref(attackOptions[currentAttackArgs.attack_type]);

// 更新攻击策略选项
function updateAttackRecipe() {
  attackRecipes.value = attackOptions[currentAttackArgs.attack_type];
}

// 处理文件夹上传
function handleFolderUpload(event: Event, type: 'model' | 'tokenizer' | 'dataset') {
  const files = (event.target as HTMLInputElement).files;
  if (files) {
    const fileNames = Array.from(files).map(file => file.name).join(', ');
    if (type === 'model') {
      currentAttackArgs.model_name_or_path = fileNames; 
    } else if (type === 'tokenizer') {
      currentAttackArgs.tokenizer_name_or_path = fileNames; 
    } else if (type === 'dataset') {
      currentAttackArgs.dataset_name_or_path = fileNames; 
    }
  }
}

// 添加攻击配置
function addAttackArgs() {
  currentAttackArgs.index = attackList.value.length + 1;
  attackList.value.push({ ...currentAttackArgs });
  Object.assign(currentAttackArgs, defaultAttackArgs());
  showAddModal.value = false;
}

// 取消添加
function cancelAddModal() {
  Object.assign(currentAttackArgs, defaultAttackArgs());
  showAddModal.value = false;
}

// 删除攻击配置
function removeAttackArgs(index: number) {
  attackList.value.splice(index, 1);
  attackList.value.forEach((item, i) => (item.index = i + 1));
}

// 编辑攻击配置
function editAttackArgs(index: number) {
  Object.assign(currentAttackArgs, attackList.value[index]);
  removeAttackArgs(index);
  showAddModal.value = true;
}

// 发送攻击配置列表
async function sendAttackList() {
  try {
    const response = await axios.post('http://localhost:5000/api/data', { attack_list: attackList.value });
    message.success('Data sent successfully!');
    console.log('Response:', response.data);
  } catch (error) {
    console.error('Failed to send data:', error);
    message.error('Failed to send data');
  }
}

// 执行攻击
async function executeAttack() {
  try {
    const response = await axios.post('http://localhost:5000/api/execute_attack');
    message.success('Attack executed successfully!');
    console.log('Attack execution response:', response.data);
  } catch (error) {
    console.error('Failed to execute attack:', error);
    message.error('Failed to execute attack');
  }
}
</script>

<style scoped>
.attack-configuration {
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 8px;
}

.header {
  display: flex;
  justify-content: flex-start;
  margin-bottom: 20px;
}

.input-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.input-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  white-space: nowrap;
  margin: 0;
  font-weight: bold;
}

.attack-table {
  margin-top: 20px;
  border-radius: 8px;
}

.action-buttons {
  display: flex;
  gap: 10px;
  margin-top: 10px;
  justify-content: flex-start;
}

.advanced-settings {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.setting-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.upload-success {
  color: green;
  margin-left: 10px;
}

.file-input {
  border-radius: 8px;
  border: 1px solid #d9d9d9;
  padding: 8px;
  width: 60%;
  transition: border-color 0.3s;
}

.file-input:focus {
  border-color: #40a9ff;
  outline: none;
}

.rounded-input {
  border-radius: 8px;
  border: 1px solid #d9d9d9;
  padding: 8px;
  width: 60%;
  transition: border-color 0.3s;
}

.rounded-input:focus {
  border-color: #40a9ff;
  outline: none;
}
</style>
