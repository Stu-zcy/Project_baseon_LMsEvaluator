// Assume Vue is globally available (from CDN)
const { ref, onMounted, defineComponent } = Vue; // Using defineComponent for better structure
import axios from "../utils/axiosConfig.js";

const AttackTable = defineComponent({
    name: 'AttackTable',
    props: {
        targetCreateTime: {
            type: Number,
            required: true,
        }
    },

    // The <template> content goes here as a string
    template: `
<div>
    <a-table v-if="responseData.AdversarialAttack.length > 0" v-bind="$attrs" 
        :columns="AdvColumns" :dataSource="responseData.AdversarialAttack" 
        :pagination="false" class="outer-table">
									<template #headerCell="{ column }">
            <template v-if="column.dataIndex === 'index'">
                <span style="display: inline-flex; align-items: center; margin-top: 5px">
                    <i class="material-icons" style=" font-size: 24px;">list</i>
                    <span>实验配置</span>
                </span>
            </template>

						<template v-else>
            {{ column.title }}
						</template>
        	</template>
        <template #title>
            <div class="flex justify-between pr-4">
                <h4>对抗攻击</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
                    实验配置{{ record[0] && record[0][0] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'success'">
                <div class="text-subtext">
                    {{ record[0] && record[0][1] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'fail'">
                <div class="text-subtext">
                    {{ record[0] && record[0][2] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'skip'">
                <div class="text-subtext">
                    {{ record[0] && record[0][3] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'before'">
                <div class="text-subtext">
                    {{ record[0] && record[0][4] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'after'">
                <div class="text-subtext">
                    {{ record[0] && record[0][5] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'rate'">
                <div class="text-subtext">
                    {{ record[0] && record[0][6] }}
                </div>
            </div>
            <div v-else class="text-subtext">
                {{ record[column.dataIndex] }}
            </div>
        </template>
    </a-table>

    <a-table v-if="responseData.PoisoningAttack.length > 0" v-bind="$attrs" 
        :columns="PoisoningColumns" :dataSource="responseData.PoisoningAttack" 
        :pagination="false" class="outer-table">
									<template #headerCell="{ column }">
            <template v-if="column.dataIndex === 'index'">
                <span style="display: inline-flex; align-items: center; margin-top: 5px">
                    <i class="material-icons" style=" font-size: 24px;">list</i>
                    <span>实验配置</span>
                </span>
            </template>

						<template v-else>
            {{ column.title }}
						</template>
        	</template>
        <template #title>
            <div class="flex justify-between pr-4">
                <h4>数据投毒</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
                    实验配置{{ record[0] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'before'">
                <div class="text-subtext">
                    {{ record[1] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'after'">
                <div class="text-subtext">
                    {{ record[2] }}
                </div>
            </div>
            <div v-else class="text-subtext">
                {{ record[column.dataIndex] }}
            </div>
        </template>
    </a-table>

    <a-table v-if="responseData.BackDoorAttack.length > 0" v-bind="$attrs" 
        :columns="BackDoorColumns" :dataSource="responseData.BackDoorAttack" 
        :pagination="false" class="outer-table">
									<template #headerCell="{ column }">
            <template v-if="column.dataIndex === 'index'">
                <span style="display: inline-flex; align-items: center; margin-top: 5px">
                    <i class="material-icons" style=" font-size: 24px;">list</i>
                    <span>实验配置</span>
                </span>
            </template>

						<template v-else>
            {{ column.title }}
						</template>
        	</template>
        <template #title>
            <div class="flex justify-between pr-4">
                <h4>后门攻击</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
                    实验配置{{ record[0] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'dataset'">
                <div class="text-subtext">
                    {{ record[1] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'Poisoner'">
                <div class="text-subtext">
                    {{ record[2] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'before'">
                <div class="text-subtext">
                    {{ record[3] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'after'">
                <div class="text-subtext">
                    {{ record[4] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'PPL'">
                <div class="text-subtext">
                    {{ record[5] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'USE'">
                <div class="text-subtext">
                    {{ record[6] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'GRAMMAR'">
                <div class="text-subtext">
                    {{ record[7] }}
                </div>
            </div>
            <div v-else class="text-subtext">
                {{ record[column.dataIndex] }}
            </div>
        </template>
    </a-table>

    <a-table v-if="responseData.RLMI.length > 0" v-bind="$attrs" 
        :columns="RLMIAttackColumns" :dataSource="responseData.RLMI" 
        :pagination="false" class="outer-table">
									<template #headerCell="{ column }">
            <template v-if="column.dataIndex === 'index'">
                <span style="display: inline-flex; align-items: center; margin-top: 5px">
                    <i class="material-icons" style=" font-size: 24px;">list</i>
                    <span>实验配置</span>
                </span>
            </template>

						<template v-else>
            {{ column.title }}
						</template>
        	</template>
        <template #title>
            <div class="flex justify-between pr-4">
                <h4>模型反演</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
                    实验配置{{ record[0] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'ASR_Attack'">
                <div class="text-subtext">
                    {{ record[1] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'WER_Attack'">
                <div class="text-subtext">
                    {{ record[2] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'ASR_Inference'">
                <div class="text-subtext">
                    {{ record[3] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'WER_Inference'">
                <div class="text-subtext">
                    {{ record[4] }}
                </div>
            </div>
            <div v-else class="text-subtext">
                {{ record[column.dataIndex] }}
            </div>
        </template>
    </a-table>

    <a-table v-if="responseData.SWAT.length > 0" v-bind="$attrs" 
        :columns="SWATAttackColumns" :dataSource="responseData.SWAT" 
        :pagination="false" @expand="getInnerData" class="outer-table">
									<template #headerCell="{ column }">
            <template v-if="column.dataIndex === 'index'">
                <span style="display: inline-flex; align-items: center; margin-top: 5px">
                    <i class="material-icons" style=" font-size: 24px;">list</i>
                    <span>实验配置</span>
                </span>
            </template>

						<template v-else-if="typeof column.title === 'string'">
            {{ column.title }}
						</template>
        	</template>
        <template #title>
            <div class="flex justify-between pr-4">
                <h4>梯度反演</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
                    实验配置{{ record.at(-1) && record.at(-1)[0] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'rouge1'">
                <div class="text-subtext">
                    {{ record.at(-1) && record.at(-1)[1] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'rouge2'">
                <div class="text-subtext">
                    {{ record.at(-1) && record.at(-1)[2] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'rougeL'">
                <div class="text-subtext">
                    {{ record.at(-1) && record.at(-1)[3] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'wrr'">
                <div class="text-subtext">
                    {{ record.at(-1) && record.at(-1)[4] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'distance'">
                <div class="text-subtext">
                    {{ record.at(-1) && record.at(-1)[5] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'fr'">
                <div class="text-subtext">
                    {{ record.at(-1) && record.at(-1)[6] }}
                </div>
            </div>
            <div v-else class="text-subtext">
                {{ record[column.dataIndex] }}
            </div>
        </template>
        <template #expandedRowRender>
            <a-table v-if="responseData.SWAT.length > 0"
                :columns="SWATInnoColumns" :data-source="SWATInnerData" 
                :pagination="false" class="inner-table">
                <template #bodyCell="{ column, record }">
                    <div class="" v-if="column.dataIndex === 'rouge1'">
                        <div class="text-subtext">
                            {{ record[2] }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'rouge2'">
                        <div class="text-subtext">
                            {{ record[3] }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'rougeL'">
                        <div class="text-subtext">
                            {{ record[4] }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'wrr'">
                        <div class="text-subtext">
                            {{ record[5] }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'distance'">
                        <div class="text-subtext">
                            {{ record[6] }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'fr'">
                        <div class="text-subtext">
                            {{ record[7] }}
                        </div>
                    </div>
                    <div v-else class="text-subtext">
                        {{ record[column.dataIndex] }}
                    </div>
                </template>
            </a-table>
        </template>
    </a-table>

    <a-table v-if="responseData.ModelStealingAttack.length > 0" v-bind="$attrs" 
        :columns="ModelStealingAttackColumns" :dataSource="responseData.ModelStealingAttack" 
        :pagination="false" class="outer-table">
					<template #headerCell="{ column }">
            <template v-if="column.dataIndex === 'index'">
                <span style="display: inline-flex; align-items: center; margin-top: 5px">
                    <i class="material-icons" style=" font-size: 24px;">list</i>
                    <span>实验配置</span>
                </span>
            </template>

						<template v-else>
            {{ column.title }}
						</template>
        	</template>
        	<template #title>
            <div class="flex justify-between pr-4">
                <h4>模型窃取攻击</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
                    实验配置{{ record.index }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'train_loss'">
                <div class="text-subtext">
                    {{ record.iteration.at(-1)[0] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'train_acc'">
                <div class="text-subtext">
                    {{ record.iteration.at(-1)[1] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'valid_acc'">
                <div class="text-subtext">
                    {{ record.iteration.at(-1)[2] }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'victim_acc'">
                <div class="text-subtext">
                    {{ record.victim_acc }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'steal_acc'">
                <div class="text-subtext">
                    {{ record.steal_acc }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'agreement'">
                <div class="text-subtext">
                    {{ record.agreement }}
                </div>
            </div>
        </template>
    </a-table>
</div>
    `,

    setup(props) { // `props` is automatically passed here
        // Constants for columns
				const AdvColumns = [
						{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
						{ title: '成功攻击次数\n(Successful Attacks，次)', dataIndex: 'success', align: 'center' },
						{ title: '失败攻击次数\n(Failed Attacks，次)', dataIndex: 'fail', align: 'center' },
						{ title: '跳过攻击次数\n(Skipped Attacks，次)', dataIndex: 'skip', align: 'center' },
						{ title: '攻击前准确率\n(Clean Accuracy，%)', dataIndex: 'before', align: 'center' },
						{ title: '攻击后准确率\n(Adversarial Accuracy，%)', dataIndex: 'after', align: 'center' },
						{ title: '攻击成功率\n(Attack Success Rate，%)', dataIndex: 'rate', align: 'center' },
				];
				const PoisoningColumns = [
						{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
						{ title: '攻击前准确率\n(Clean Accuracy，%)', dataIndex: 'before', align: 'center' },
						{ title: '攻击后准确率\n(Adversarial Accuracy，%)', dataIndex: 'after', align: 'center' },
				];
				const BackDoorColumns = [
						{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
						{ title: '投毒方数据集\n(Poisoned Dataset，名称)', dataIndex: 'dataset', align: 'center' },
						// { title: '投毒方', dataIndex: 'Poisoner', align: 'center' },
						{ title: '原始数据集准确率\n(Clean Accuracy，%)', dataIndex: 'before', align: 'center' },
						{ title: '毒化数据集准确率\n(Poisoned Accuracy，%)', dataIndex: 'after', align: 'center' },
						{ title: '困惑度\n(PPL-Perplexity)', dataIndex: 'PPL', align: 'center' },
						{ title: '语义相似性\n(USE-Universal Sentence Encoder Similarity)', dataIndex: 'USE', align: 'center' },
						{ title: '语法正确性得分\n(Grammar Score)', dataIndex: 'GRAMMAR', align: 'center' },
				];
				const RLMIAttackColumns = [
						{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
						{ title: '攻击阶段成功率\n(Attack Success Rate (Attack Phase)，%)', dataIndex: 'ASR_Attack', align: 'center' },
						{ title: '攻击阶段词错误率\n(Word Error Rate (Attack Phase)，%)', dataIndex: 'WER_Attack', align: 'center' },
						{ title: '推理阶段成功率\n(Attack Success Rate (Inference Phase)，%)', dataIndex: 'ASR_Inference', align: 'center' },
						{ title: '推理阶段词错误率\n(Word Error Rate (Inference Phase)，%)', dataIndex: 'WER_Inference', align: 'center' },
				];
				const SWATAttackColumns = [
						{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
						{ title: 'ROUGE-1得分\n(ROUGE-1 Score，%)', dataIndex: 'rouge1', align: 'center' },
						{ title: 'ROUGE-2得分\n(ROUGE-2 Score，%)', dataIndex: 'rouge2', align: 'center' },
						{ title: 'ROUGE-L得分\n(ROUGE-L Score，%)', dataIndex: 'rougeL', align: 'center' },
						{ title: '词汇恢复率\n(Token Recovery Rate，%)', dataIndex: 'wrr', align: 'center' },
						{ title: '编辑距离\n(Edit Distance)', dataIndex: 'distance', align: 'center' },
						{ title: '完全恢复率\n(Exact Match Rate，%)', dataIndex: 'fr', align: 'center' },
				];
				const SWATInnoColumns = [
						{ title: 'ROUGE-1得分\n(ROUGE-1 Score，%)', dataIndex: 'rouge1', align: 'center' },
						{ title: 'ROUGE-2得分\n(ROUGE-2 Score，%)', dataIndex: 'rouge2', align: 'center' },
						{ title: 'ROUGE-L得分\n(ROUGE-L Score，%)', dataIndex: 'rougeL', align: 'center' },
						{ title: '词汇恢复率\n(Token Recovery Rate，%)', dataIndex: 'wrr', align: 'center' },
						{ title: '编辑距离\n(Edit Distance)', dataIndex: 'distance', align: 'center' },
						{ title: '完全恢复', dataIndex: 'fr', align: 'center' },
				];
				const ModelStealingAttackColumns = [
						{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
						{ title: '训练集准确率\n(Training Accuracy，%)', dataIndex: 'train_acc', align: 'center' },
						{ title: '训练损失\n(Training Loss)', dataIndex: 'train_loss', align: 'center' },
						{ title: '验证集准确率\n(Validation Accuracy，%)', dataIndex: 'valid_acc', align: 'center' },
						{ title: '目标模型准确率\n(Target Model Accuracy，%)', dataIndex: 'victim_acc', align: 'center' },
						{ title: '替代模型准确率\n(Stolen Model Accuracy，%)', dataIndex: 'steal_acc', align: 'center' },
						{ title: '一致性\n(Agreement，%)', dataIndex: 'agreement', align: 'center' },
				];

        const responseData = ref({
            AdversarialAttack: [],
            BackDoorAttack: [],
            PoisoningAttack: [],
            RLMI: [],
            SWAT: [],
            ModelStealingAttack: []
        });

        // 子表格
        const SWATInnerData = ref([]);

        const username = localStorage.getItem('Global_username');
        const token = localStorage.getItem('Global_token');

        function getInnerData(expanded, record) {
            if (expanded) {
                SWATInnerData.value = record.slice(0, -1); // Exclude the last summary row
                console.log("Expanded SWAT inner data:", SWATInnerData.value);
            }
        }

        onMounted(async () => {
            if (!props.targetCreateTime) {
                console.warn("targetCreateTime prop is not provided or is invalid.");
                // Potentially set responseData to empty or show an error message
                // For now, we'll just not make the API call.
                return;
            }
            try {
                // Ensure axios is available globally if not imported via a module system
                if (typeof axios === 'undefined') {
                    console.error("axios is not defined. Make sure it's loaded globally via CDN.");
                    return;
                }
                const response = await axios.post('/api/getRecord', {
                    username: username,
                    token: token,
                    createTime: props.targetCreateTime
                });
                // It's safer to check if response.data exists and is an object
                if (response && response.data && typeof response.data === 'object') {
                    // Assign specific keys if they exist, or default to empty arrays
                    responseData.value.AdversarialAttack = response.data.AdversarialAttack || [];
                    responseData.value.BackDoorAttack = response.data.BackDoorAttack || [];
                    responseData.value.PoisoningAttack = response.data.PoisoningAttack || [];
                    responseData.value.RLMI = response.data.RLMI || [];
                    responseData.value.SWAT = response.data.SWAT || [];
                    responseData.value.ModelStealingAttack = response.data.ModelStealingAttack || [];
                } else {
                    console.error("Invalid API response structure:", response);
                }
                console.log("Fetched responseData:", responseData.value);
            } catch (error) {
                console.error("请求出错 (Error fetching data):", error);
                // Reset data or show error message to user
                responseData.value = {
                    AdversarialAttack: [],
                    BackDoorAttack: [],
                    PoisoningAttack: [],
                    RLMI: [],
                    SWAT: [],
                    ModelStealingAttack: []
                };
            }

            // 预处理各个结果，添加索引什么的
            if (responseData.value.AdversarialAttack && Array.isArray(responseData.value.AdversarialAttack)) {
                responseData.value.AdversarialAttack = responseData.value.AdversarialAttack.map((item, index) => {
                    return [[index + 1, ...item]];
                });
            }
            if (responseData.value.PoisoningAttack && Array.isArray(responseData.value.PoisoningAttack)) {
                responseData.value.PoisoningAttack = responseData.value.PoisoningAttack.map((item, index) => {
                    return [index + 1, ...item]; // This creates [index, item_val1, item_val2, ...]
                });
            }
            if (responseData.value.BackDoorAttack && Array.isArray(responseData.value.BackDoorAttack)) {
                responseData.value.BackDoorAttack = responseData.value.BackDoorAttack.map((item, index) => {
                    return [index + 1, ...item];
                });
            }
            if (responseData.value.RLMI && Array.isArray(responseData.value.RLMI)) {
                responseData.value.RLMI = responseData.value.RLMI.map((item, index) => {
                    return [index + 1, ...item];
                });
            }
            if (responseData.value.SWAT && Array.isArray(responseData.value.SWAT)) {
                responseData.value.SWAT = responseData.value.SWAT.map((item, index) => {
                    return item.map((element) => {
                        if (element.length == 6) {
                            return [index + 1, ...element]
                        } else {
                            return element
                        }
                    })
                })
            }
            if (responseData.value.ModelStealingAttack && Array.isArray(responseData.value.ModelStealingAttack)) {
                responseData.value.ModelStealingAttack = responseData.value.ModelStealingAttack.map((item, index) => {
                    item.index = index + 1; 
                    return item; 
                });
            }
            console.log("Processed responseData:", responseData.value);

        });

        // Return everything that the template needs
        return {
            AdvColumns,
            PoisoningColumns,
            BackDoorColumns,
            RLMIAttackColumns,
            SWATAttackColumns,
            SWATInnoColumns,
            ModelStealingAttackColumns,
            responseData,
            SWATInnerData,
            getInnerData,
            // username, token, props are available via `this` or closure, but not directly needed by template
        };
    }
});

export default AttackTable;