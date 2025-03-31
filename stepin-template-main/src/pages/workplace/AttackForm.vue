<template>
  <div class="configuration">
    <!-- 攻击配置部分 -->
    <div class="attack-configuration">
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

      <a-modal v-model:visible="showAddModal" title="添加攻击配置" @ok="addAttackArgs" @cancel="cancelAddModal" width="650px">
        <div class="input-section">

          <!-- 攻击类型选择框 -->
          <label class="input-label">
            攻击类型:
            <a-select
              v-model:value="currentAttackArgs.attack_type"
              class="select-input"
              :options="attack_type_Data.map(type => ({ value: type, label: type }))"
              placeholder="请选择攻击类型"
            />
          </label>

          <!-- 攻击策略选择框 -->
          <label class="input-label">
            攻击策略:
            <a-select
              v-model:value="currentAttackArgs.attack_recipe"
              class="select-input"
              :options="attack_recipes.map(recipe => ({ value: recipe, label: recipe }))"
              placeholder="请选择攻击策略"
            />
          </label>

          <!-- 高级设置折叠面板 -->
          <a-collapse v-if="currentAttackArgs.attack_type" defaultActiveKey="1">
            <a-collapse-panel header="高级设置" key="1">
              <div v-if="currentAttackArgs.attack_type === 'AdversarialAttack'">
                使用本地模型 <a-switch v-model:checked="currentAttackArgs.use_local_model" /> 
                使用本地分词器<a-switch v-model:checked="currentAttackArgs.use_local_tokenizer" />
                使用本地数据集<a-switch v-model:checked="currentAttackArgs.use_local_dataset" /> 
                攻击次数<a-input-number v-model:value="currentAttackArgs.attack_nums" class="rounded-input" :min="1" /> 
              </div>

              <div v-if="currentAttackArgs.attack_type === 'FET'">
                随机种子<a-input-number v-model:value="currentAttackArgs.seed" class="rounded-input" /> 
                攻击批次<a-input-number v-model:value="currentAttackArgs.attack_batch" class="rounded-input" :min="1" />
                攻击次数<a-input-number v-model:value="currentAttackArgs.attack_nums" class="rounded-input" :min="1" />
                <a-select v-model:value="currentAttackArgs.distance_func" class="select-input" :options="['l2', 'cos'].map(func => ({ value: func, label: func }))" placeholder="选择距离函数" />
                种群大小<a-input-number v-model:value="currentAttackArgs.population_size" class="rounded-input" /> 
                选择批次<a-input-number v-model:value="currentAttackArgs.tournsize" class="rounded-input" /> 
                交叉率<a-input-number v-model:value="currentAttackArgs.crossover_rate" class="rounded-input" :step="0.1" :min="0" :max="1" /> 
                突变率<a-input-number v-model:value="currentAttackArgs.mutation_rate" class="rounded-input" :step="0.1" :min="0" :max="1" /> 
                最大代数<a-input-number v-model:value="currentAttackArgs.max_generations" class="rounded-input" /> 
                精英个体大小<a-input-number v-model:value="currentAttackArgs.halloffame_size" class="rounded-input" />
              </div>

              <div v-if="currentAttackArgs.attack_type === 'BackDoorAttack'">
                <a-checkbox-group v-model:value="currentAttackArgs.sample_metrics">
                  <a-checkbox value="ppl">PPL</a-checkbox>
                  <a-checkbox value="use">USE</a-checkbox>
                  <a-checkbox value="grammar">Grammar</a-checkbox>
                </a-checkbox-group>
                <a-select v-model:value="currentAttackArgs.defender" class="select-input" :options="['BKI', 'ONION', 'STRIP', 'RAP', 'CUBE'].map(defender => ({ value: defender, label: defender }))" placeholder="选择防御方法" />
              </div>

              <div v-if="currentAttackArgs.attack_type === 'PoisoningAttack'">
                投毒率<a-input-number v-model:value="currentAttackArgs.poisoning_rate" class="rounded-input" :min="0" :max="1" step="0.01" /> 
                训练周期<a-input-number v-model:value="currentAttackArgs.epochs" class="rounded-input" /> 
              </div>

              <div v-if="currentAttackArgs.attack_type === 'RLMI'">
                随机种子<a-input-number v-model:value="currentAttackArgs.seed" class="rounded-input" /> 
                序列长度<a-input-number v-model:value="currentAttackArgs.seq_length" class="rounded-input" /> 
                目标标签<a-input-number v-model:value="currentAttackArgs.target_label" class="rounded-input" /> 
                最大迭代次数<a-input-number v-model:value="currentAttackArgs.max_iterations" class="rounded-input" /> 
                最小输入长度<a-input-number v-model:value="currentAttackArgs.min_input_length" class="rounded-input" />
                最大输入长度<a-input-number v-model:value="currentAttackArgs.max_input_length" class="rounded-input" />
                生成数量<a-input-number v-model:value="currentAttackArgs.num_generation" class="rounded-input" /> 
              </div>

              <div v-if="currentAttackArgs.attack_type === 'GIAforNLP'">
                <a-input v-model:value="currentAttackArgs.optimizer" class="rounded-input" placeholder="优化器" />
                攻击批次<a-input-number v-model:value="currentAttackArgs.attack_batch" class="rounded-input" /> 
                攻击次数<a-input-number v-model:value="currentAttackArgs.attack_nums" class="rounded-input" /> 
                <a-select v-model:value="currentAttackArgs.distance_func" class="select-input" :options="['l2', 'cos'].map(func => ({ value: func, label: func }))" placeholder="选择距离函数" />
                学习率<a-input-number v-model:value="currentAttackArgs.attack_lr" class="rounded-input" step="0.01" /> 
                迭代次数<a-input-number v-model:value="currentAttackArgs.attack_iters" class="rounded-input" /> 
              </div>
            </a-collapse-panel>
          </a-collapse>
        </div>
      </a-modal>

      <div class="action-buttons">
        <a-button type="primary" @click="sendAttackList" class="action-button">发送攻击配置</a-button>
        <a-button type="default" @click="executeAttack" class="action-button">执行攻击</a-button>
        <!-- 运行中弹框 -->
        <a-modal v-model:visible="isModalVisible" title="攻击执行中" :footer="null">
          <div class="modal-content">
            <a-spin size="large" tip="攻击执行中..."></a-spin>
            <p class="loading-text">正在处理中，请稍候...</p>
          </div>
        </a-modal>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, computed, watch } from 'vue';
import axios from 'axios';
import { message } from 'ant-design-vue';

// 攻击类型数据和对应的攻击策略数据
const attack_type_Data = ['AdversarialAttack', 'FET', 'BackDoorAttack', 'PoisoningAttack', 'RLMI', 'GIAforNLP'];

const attack_recipe_Data = {
  GIAforNLP: ['default'],
  RLMI: ['default'],
  AdversarialAttack: [
    'A2TYoo2021', 'BAEGarg2019', 'BERTAttackLi2020', 'GeneticAlgorithmAlzantot2018',
    'FasterGeneticAlgorithmJia2019', 'DeepWordBugGao2018', 'HotFlipEbrahimi2017',
    'InputReductionFeng2018', 'Kuleshov2017', 'MorpheusTan2020', 'Seq2SickCheng2018BlackBox',
    'TextBuggerLi2018', 'TextFoolerJin2019', 'PWWSRen2019', 'IGAWang2019', 'Pruthi2019',
    'PSOZang2020', 'CheckList2020', 'CLARE2020', 'FrenchRecipe', 'SpanishRecipe', 'ChineseRecipe'
  ],
  BackDoorAttack: ['BadNets','AddSent','SynBkd','StyleBkd','POR','TrojanLM','SOS','LWP','EP','NeuBA','LWS','RIPPLES'],
  PoisoningAttack: ['default'],
  FET:['default']
};

const defaultAttackConfig = {
  AdversarialAttack: {
    attack_type: 'AdversarialAttack',
    attack_recipe: 'TextFoolerJin2019',
    use_local_model: true,
    use_local_tokenizer: true,
    use_local_dataset: true,
    attack_nums: 2,
    display_full_info: true,
  },
  FET: {
    attack_type: 'FET',
    attack_recipe: 'default',
    seed: 42,
    attack_batch: 2,
    attack_nums: 1,
    distance_func: 'l2',
    population_size: 300,
    tournsize: 5,
    crossover_rate: 0.9,
    mutation_rate: 0.1,
    max_generations: 2,
    halloffame_size: 30,
    display_full_info: true,
  },
  BackDoorAttack: {
    attack_type: 'BackDoorAttack',
    attack_recipe: 'default',
    sample_metrics: [],
    display_full_info: true,
    defender: 'None',
  },
  PoisoningAttack: {
    attack_type: 'PoisoningAttack',
    attack_recipe: 'default',
    poisoning_rate: 0.1,
    epochs: 10,
    display_full_info: true,
  },
  RLMI: {
    attack_type: 'RLMI',
    attack_recipe: 'default',
    seed: 42,
    seq_length: 20,
    target_label: 0,
    max_iterations: 2000,
    min_input_length: 2,
    max_input_length: 5,
    num_generation: 1000,
  },
  GIAforNLP: {
    attack_type: 'GIAforNLP',
    attack_recipe: 'default',
    optimizer: 'Adam',
    attack_batch: 2,
    attack_nums: 1,
    distance_func: 'l2',
    attack_lr: 0.01,
    attack_iters: 10,
    display_full_info: true,
  },
};

// 当前攻击配置
const currentAttackArgs = reactive(defaultAttackConfig['AdversarialAttack']);

// 计算属性：根据选择的攻击类型动态更新攻击策略选项
const attack_recipes = computed(() => currentAttackArgs.attack_type ? attack_recipe_Data[currentAttackArgs.attack_type] : []);

// 监听攻击类型变化，自动更新攻击策略和高级配置
watch(() => currentAttackArgs.attack_type, (newAttackType) => {
  Object.assign(currentAttackArgs, defaultAttackConfig[newAttackType]);
  if (newAttackType) {
    currentAttackArgs.attack_recipe = attack_recipe_Data[newAttackType][0];
  } else {
    currentAttackArgs.attack_recipe = '';
  }
});

// 需要添加的变量
const isModalVisible = ref(false);  // 缺少的定义

const columns = [
  { title: '序号', dataIndex: 'index' },
  { title: 'Attack Type', dataIndex: 'attack_type' },
  { title: 'Attack Recipe', dataIndex: 'attack_recipe' },
  { title: '操作', dataIndex: 'operation' },
];

const attackList = ref([]);
const showAddModal = ref(false);

function addAttackArgs() {
  const attackConfigCopy = { ...currentAttackArgs };
  attackList.value.push(attackConfigCopy);
  showAddModal.value = false;
}

function cancelAddModal() {
  Object.assign(currentAttackArgs, defaultAttackConfig['AdversarialAttack']);
  showAddModal.value = false;
}

function removeAttackArgs(index: number) {
  attackList.value.splice(index, 1);
  attackList.value.forEach((item, i) => (item.index = i + 1));
}

function editAttackArgs(index: number) {
  Object.assign(currentAttackArgs, attackList.value[index]);
  removeAttackArgs(index);
  showAddModal.value = true;
}

async function sendAttackList() {
  const username = localStorage.getItem('Global_username');
  const token = localStorage.getItem('Global_token'); 
  try {
    const response = await axios.post('http://127.0.0.1:5000/api/attack_list', {
      attack_list: attackList.value, 
      username: username,
      token: token
    });
    message.success('攻击配置发送成功！');
    console.log('Response:', response.data);
  } catch (error) {
    console.error('Failed to send attack data:', error);
    message.error('攻击配置发送失败');
  }
}

async function executeAttack() {
  const username = localStorage.getItem('Global_username');
  const token = localStorage.getItem('Global_token');
  
  if (!username) {
    message.error('未找到用户名，请重新登录');
    return;
  }

  isModalVisible.value = true;

  try {
    const response = await axios.post('http://127.0.0.1:5000/api/execute_attack', {
      username: username,
      token: token
    });
    message.success('攻击执行成功！');
    console.log('Attack execution response:', response.data);
  } catch (error) {
    console.error('Failed to execute attack:', error);
    message.error('攻击执行失败');
  } finally {
    isModalVisible.value = false;
  }
}
</script>


<style scoped>
.configuration {
  padding: 30px;
  font-family: 'Arial', sans-serif;
  background-color: #f7f7f7;
}

.attack-configuration {
  margin-bottom: 40px;
}

.header {
  display: flex;
  justify-content: flex-start;
  margin-bottom: 20px;
}

.input-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.input-label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  white-space: nowrap;
  margin: 0;
  font-weight: bold;
}

.select-input {
  width: 100%;
  border-radius: 8px;
  padding: 10px;
  border: 1px solid #d9d9d9;
  font-size: 14px;
}

.rounded-input {
  border-radius: 8px;
  border: 1px solid #d9d9d9;
  padding: 10px;
  font-size: 14px;
}

.attack-table {
  margin-top: 20px;
  border-radius: 8px;
  background-color: #ffffff;
}

.action-buttons {
  display: flex;
  gap: 10px;
  margin-top: 20px;
  justify-content: flex-start;
}

.action-button {
  font-weight: bold;
  padding: 10px 20px;
}

.modal-content {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.loading-text {
  font-size: 16px;
  font-weight: 500;
  margin-top: 10px;
}

</style>