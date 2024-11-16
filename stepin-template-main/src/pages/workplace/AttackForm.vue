<template>
  <div class="attack-configuration">
    <!-- 攻击配置部分 -->
    <div class="header">
      <a-button type="primary" icon="plus" @click="showAddModal = true">添加攻击配置</a-button>
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

        <!-- 攻击类型选择框和攻击策略选择框 -->
        <label class="input-label">
          攻击类型:
          <a-select
            v-model:value="currentAttackArgs.attack_type"
            style="width: 200px"
            :options="attack_type_Data.map(type => ({ value: type, label: type }))"
          />
        </label>

        <label class="input-label">
          攻击策略:
          <a-select
            v-model:value="currentAttackArgs.attack_recipe"
            style="width: 200px"
            :options="attack_recipes.map(recipe => ({ value: recipe, label: recipe }))"
          />
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

  <!-- 防御配置部分 -->
  <div class="defense-configuration">
    <div class="header">
      <a-button type="primary" icon="plus" @click="showDefenseModal = true">添加防御配置</a-button>
    </div>

    <a-table :columns="defenseColumns" :dataSource="defenseList" rowKey="index" class="defense-table">
      <template #bodyCell="{ column, record, index }">
        <span v-if="column.dataIndex === 'operation'">
          <a-button type="link" @click="editDefenseArgs(index)">编辑</a-button>
          <a-button type="link" danger @click="removeDefenseArgs(index)">删除</a-button>
        </span>
      </template>
    </a-table>

    <a-modal v-model:visible="showDefenseModal" title="添加防御配置" @ok="addDefenseArgs" @cancel="cancelDefenseModal" width="600px">
      <div class="input-section">
        <label class="input-label">
          防御类型:
          <a-select v-model:value="currentDefenseArgs.defense_type" style="width: 200px" :options="defenseTypes" />
        </label>

        <label class="input-label">
          防御策略:
          <a-select v-model:value="currentDefenseArgs.defense_recipe" style="width: 200px" :options="defenseRecipes[currentDefenseArgs.defense_type]" />
        </label>

        <!-- 高级设置 -->
        <a-collapse>
          <a-collapse-panel header="高级设置" key="1">
            <div class="advanced-settings">
              <div class="setting-item">
                <label class="input-label">启用自定义防御模型:</label>
                <a-switch v-model:checked="currentDefenseArgs.use_custom_model" />
              </div>
              <div class="setting-item">
                <label class="input-label">自定义模型路径:</label>
                <input type="file" webkitdirectory @change="(event) => handleFolderUpload(event, 'defense_model')" class="file-input" />
                <span v-if="currentDefenseArgs.defense_model_path" class="upload-success">上传成功: {{ currentDefenseArgs.defense_model_path }}</span>
              </div>
            </div>
          </a-collapse-panel>
        </a-collapse>
      </div>
    </a-modal>

    <div class="action-buttons">
      <a-button type="primary" @click="sendDefenseList">发送所有防御配置</a-button>
      <a-button type="default" @click="executeDefense">执行防御</a-button>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { computed, ref, reactive, watch } from 'vue';
import axios from 'axios';
import { message } from 'ant-design-vue';

// 攻击类型和防御类型配置
const attack_type_Data = ['GIAforNLP', 'SWAT', 'AdvAttack', 'BackDoorAttack', 'PoisoningAttack'];
const attack_recipe_Data = {
  GIAforNLP: ['default'],
  SWAT: ['default'],
  AdvAttack: ['A2TYoo2021', 'BAEGarg2019', 'BERTAttackLi2020'],
  BackDoorAttack: ['default'],
  PoisoningAttack: ['default']
};

const defenseTypes = ['CustomDefense', 'PreprocessingDefense', 'PostprocessingDefense'];
const defenseRecipes = {
  CustomDefense: ['default'],
  PreprocessingDefense: ['default'],
  PostprocessingDefense: ['default']
};

// 默认攻击和防御配置模板
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

const defaultDefenseArgs = () => ({
  index: 0,
  defense_type: 'CustomDefense',
  defense_recipe: 'default',
  use_custom_model: false,
  defense_model_path: '',
});

// 当前攻击和防御配置
const currentAttackArgs = reactive(defaultAttackArgs());
const currentDefenseArgs = reactive(defaultDefenseArgs());

// 攻击配置列表
const attackList = ref([]);
const defenseList = ref([]);

// 表格列配置
const columns = [
  { title: '序号', dataIndex: 'index' },
  { title: 'Attack Type', dataIndex: 'attack_type' },
  { title: 'Attack Recipe', dataIndex: 'attack_recipe' },
  { title: 'Model Name or Path', dataIndex: 'model_name_or_path' },
  { title: '操作', dataIndex: 'operation' },
];

const defenseColumns = [
  { title: '序号', dataIndex: 'index' },
  { title: 'Defense Type', dataIndex: 'defense_type' },
  { title: 'Defense Recipe', dataIndex: 'defense_recipe' },
  { title: 'Defense Model Path', dataIndex: 'defense_model_path' },
  { title: '操作', dataIndex: 'operation' },
];

// 显示添加攻击配置的模态框
const showAddModal = ref(false);

// 显示添加防御配置的模态框
const showDefenseModal = ref(false);

// 添加攻击配置
function addAttackArgs() {
  currentAttackArgs.index = attackList.value.length + 1;
  attackList.value.push({ ...currentAttackArgs });
  Object.assign(currentAttackArgs, defaultAttackArgs());
  showAddModal.value = false;
}

// 取消添加攻击配置
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

// 添加防御配置
function addDefenseArgs() {
  currentDefenseArgs.index = defenseList.value.length + 1;
  defenseList.value.push({ ...currentDefenseArgs });
  Object.assign(currentDefenseArgs, defaultDefenseArgs());
  showDefenseModal.value = false;
}

// 取消添加防御配置
function cancelDefenseModal() {
  Object.assign(currentDefenseArgs, defaultDefenseArgs());
  showDefenseModal.value = false;
}

// 删除防御配置
function removeDefenseArgs(index: number) {
  defenseList.value.splice(index, 1);
  defenseList.value.forEach((item, i) => (item.index = i + 1));
}

// 编辑防御配置
function editDefenseArgs(index: number) {
  Object.assign(currentDefenseArgs, defenseList.value[index]);
  removeDefenseArgs(index);
  showDefenseModal.value = true;
}

// 发送攻击配置列表
async function sendAttackList() {
  const username = localStorage.getItem('Global_username');
  
  try {
    const response = await axios.post('http://localhost:5000/api/attack_List', { 
      attack_list: attackList.value, 
      username: username
    });
    message.success('数据发送成功！');
    console.log('Response:', response.data);
  } catch (error) {
    console.error('Failed to send data:', error);
    message.error('数据发送失败');
  }
}

// 执行攻击
async function executeAttack() {
  const username = localStorage.getItem('Global_username');
  
  if (!username) {
    message.error('未找到用户名，请重新登录');
    return;
  }
  
  try {
    const response = await axios.post('http://localhost:5000/api/execute_attack', { 
      username: username
    });
    message.success('攻击执行成功！');
    console.log('Attack execution response:', response.data);
  } catch (error) {
    console.error('Failed to execute attack:', error);
    message.error('攻击执行失败');
  }
}

// 发送防御配置列表
async function sendDefenseList() {
  const username = localStorage.getItem('Global_username');
  
  try {
    const response = await axios.post('http://localhost:5000/api/defense_list', { 
      defense_list: defenseList.value, 
      username: username
    });
    message.success('防御配置发送成功！');
    console.log('Response:', response.data);
  } catch (error) {
    console.error('Failed to send defense data:', error);
    message.error('防御配置发送失败');
  }
}

// 执行防御
async function executeDefense() {
  const username = localStorage.getItem('Global_username');
  
  if (!username) {
    message.error('未找到用户名，请重新登录');
    return;
  }
  
  try {
    const response = await axios.post('http://localhost:5000/api/execute_defense', { 
      username: username
    });
    message.success('防御执行成功！');
    console.log('Defense execution response:', response.data);
  } catch (error) {
    console.error('Failed to execute defense:', error);
    message.error('防御执行失败');
  }
}

</script>

<style scoped>
.attack-configuration, .defense-configuration {
  padding: 20px;
  background-color: #f9f9f9;
  border-radius: 8px;
  margin-bottom: 20px;
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

.attack-table, .defense-table {
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

.file-input, .rounded-input {
  border-radius: 8px;
  border: 1px solid #d9d9d9;
  padding: 8px;
  width: 60%;
  transition: border-color 0.3s;
}

.file-input:focus, .rounded-input:focus {
  border-color: #40a9ff;
  outline: none;
}
</style>
