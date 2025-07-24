<script lang="ts" setup>
import { ref, onMounted, withDefaults } from 'vue';
import axios from "axios"

const AdvColumns = [
	{
		title: '索引',
		dataIndex: 'index',
		customRender: ({ text, index }) => { index }
	},
	{ title: '成功攻击次数', dataIndex: 'success' },
	{ title: '失败攻击次数', dataIndex: 'fail' },
	{ title: '跳过攻击次数', dataIndex: 'skip' },
	{ title: '准确率（攻击前）', dataIndex: 'before' },
	{ title: '准确率（攻击后）', dataIndex: 'after' },
	{ title: '成功率', dataIndex: 'rate' },
];
const PoisoningColumns = [
	{
		title: '索引',
		dataIndex: 'index',
	},
	{ title: '准确率（攻击前）', dataIndex: 'before' },
	{ title: '准确率（攻击后）', dataIndex: 'after' },
];
const BackDoorColumns = [
	{ title: '索引', dataIndex: 'index' },
	{ title: '数据集', dataIndex: 'dataset' },
	{ title: '投毒方', dataIndex: 'Poisoner' },
	{ title: '准确率（原始数据集）', dataIndex: 'before' },
	{ title: '准确率（毒化数据集）', dataIndex: 'after' },
	{ title: 'PPL', dataIndex: 'PPL' },
	{ title: 'USE', dataIndex: 'USE' },
	{ title: 'GRAMMAR', dataIndex: 'GRAMMAR' },
];
const RLMIAttackColumns = [
	{ title: '索引', dataIndex: 'index'},
	{ title: 'average ASR(Attack)', dataIndex: 'ASR_Attack' },
	{ title: 'average WER(Attack)', dataIndex: 'WER_Attack' },
	{ title: 'average ASR(inference)', dataIndex: 'ASR_Inference' },
	{ title: 'average WER(inference)', dataIndex: 'WER_Inference' },
]
const FETAttackColumns = [
	{ title: '轮', dataIndex: 'index'},
	{ title: 'average rouge1', dataIndex: 'rouge1' },
	{ title: 'average rouge2', dataIndex: 'rouge2' },
	{ title: 'average rougeL', dataIndex: 'rougeL' },
	{ title: 'average Word recovery rate', dataIndex: 'wrr' },
	{ title: 'average Edit distance', dataIndex: 'distance' },
	{ title: 'full recovery rate', dataIndex: 'fr' },
];
const FETInnoColumns = [
	{ title: 'rouge1', dataIndex: 'rouge1' },
	{ title: 'rouge2', dataIndex: 'rouge2' },
	{ title: 'rougeL', dataIndex: 'rougeL' },
	{ title: 'Word recovery rate', dataIndex: 'wrr' },
	{ title: 'Edit distance', dataIndex: 'distance' },
	{ title: 'if full recovery', dataIndex: 'fr' },
]

const responseData = ref({
  AdversarialAttack: [],
	BackDoorAttack: [],
	PoisoningAttack: [],
	RLMI: [],
	FET: [],
});

const BackDoorAttack = ref([])
const FETInnerData = ref([])

const username = localStorage.getItem('Global_username');  // 从 localStorage 获取用户名
const token = localStorage.getItem('Global_token');
const props = defineProps({
	targetCreateTime: Number
})
function getInnerData(expanded, record) {
	if (expanded) {
		FETInnerData.value = record.slice(0, -1);
		console.log(record.slice(0, -1));
	}
}
onMounted(async () => {
	try {
		const response = await axios.post('http://127.0.0.1:46666/api/getRecord', {
			username: username,
			token: token,
			createTime: props.targetCreateTime
		});
		responseData.value = response.data;
		console.log(responseData.value);
	} catch (error) {
		console.error("请求出错", error);
	}

	if (responseData.value.AdversarialAttack) {
		responseData.value.AdversarialAttack = responseData.value.AdversarialAttack.map((item, index) => {
			return [[index + 1, ...item]];
		});
	}
	if (responseData.value.PoisoningAttack) {
		responseData.value.PoisoningAttack = responseData.value.PoisoningAttack.map((item, index) => {
			return [index + 1, ...item];
		});
	}
	if (responseData.value.BackDoorAttack) {
		responseData.value.BackDoorAttack = responseData.value.BackDoorAttack.map((item, index) => {
			return [index + 1, ...item];
		});
	}
	if (responseData.value.RLMI) {
		responseData.value.RLMI = responseData.value.RLMI.map((item, index) => {
			return [index + 1, ...item];
		});
	}
	if (responseData.value.FET) {
		responseData.value.FET = responseData.value.FET.map((item, index) => {
			return item.map((element) => {
				if (element.length == 6) {
					return [index+1, ...element]
				} else {
					return element
				}
			})
		})
	}
	console.log(responseData.value)

	//后门攻击的输出改为pretty table
	// BackDoorColumns = [
	// 	{
	// 		//这个格子需要考虑合并影响
	// 		title: '索引',
	// 		dataIndex: 'index',
	// 		customCell: ((record, rowIndex) => {
	// 			if (record[1] != 'epoch:0')
	// 				return { rowSpan: 0 }
	// 			// else 
	// 			// 	return {rowSpan: 4}
	// 			let count = 0
	// 			while (BackDoorAttack.value[rowIndex + count][1] != '/' && rowIndex + count < BackDoorAttack.value.length)
	// 				count++;
	// 			console.log(count)
	// 			if (rowIndex + count < BackDoorAttack.value.length)
	// 				return { rowSpan: count + 1 }
	// 			else {
	// 				console.error("缺少'/', 请检查数据是否完整'")
	// 				return
	// 			}
	// 		})
	// 	},
	// 	{ title: 'Epoach', dataIndex: 'sub_index' },
	// 	{ title: '损失函数', dataIndex: 'loss' },
	// 	{ title: '准确率（原数据集）', dataIndex: 'clean' },
	// 	{ title: '准确率（污染数据集）', dataIndex: 'poison' },
	// ]
});
</script>

<template>
	<a-table v-bind="$attrs" :columns="AdvColumns" :dataSource="responseData.AdversarialAttack" :pagination="false">
		<template #title>
			<div class="flex justify-between pr-4">
				<h4>AdversarialAttack Results</h4>
			</div>
		</template>
		<template #bodyCell="{ column, text, record }">
			<div class="" v-if="column.dataIndex === 'index'">
				<div class="text-subtext">
					{{ record[0][0] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'success'">
				<div class="text-subtext">
					{{ record[0][1] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'fail'">
				<div class="text-subtext">
					{{ record[0][2] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'skip'">
				<div class="text-subtext">
					{{ record[0][3] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'before'">
				<div class="text-subtext">
					{{ record[0][4] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'after'">
				<div class="text-subtext">
					{{ record[0][5] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rate'">
				<div class="text-subtext">
					{{ record[0][6] }}
				</div>
			</div>
			<div v-else class="text-subtext">
				{{ text }}
			</div>
		</template>
	</a-table>

	<a-table v-bind="$attrs" :columns="PoisoningColumns" :dataSource="responseData.PoisoningAttack" :pagination="false">
		<template #title>
			<div class="flex justify-between pr-4">
				<h4>PoisoningAttack Results</h4>
			</div>
		</template>
		<template #bodyCell="{ column, text, record }">
			<div class="" v-if="column.dataIndex === 'index'">
				<div class="text-subtext">
					{{ record[0] }}
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
				{{ text }}
			</div>
		</template>
	</a-table>

	<a-table v-bind="$attrs" :columns="BackDoorColumns" :dataSource="responseData.BackDoorAttack" :pagination="false">
		<template #title>
			<div class="flex justify-between pr-4">
				<h4>BackDoorAttackp Results</h4>
			</div>
		</template>
		<template #bodyCell="{ column, text, record }">
			<div class="" v-if="column.dataIndex === 'index'">
				<div class="text-subtext">
					{{ record[0] }}
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
				{{ text }}
			</div>
		</template>
	</a-table>

	<a-table v-bind="$attrs" :columns="RLMIAttackColumns" :dataSource="responseData.RLMI" :pagination="false">
		<template #title>
			<div class="flex justify-between pr-4">
				<h4>RLMI_Attack Results</h4>
			</div>
		</template>
		<template #bodyCell="{ column, text, record }">
			<div class="" v-if="column.dataIndex === 'index'">
				<div class="text-subtext">
					{{ record[0] }}
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
				{{ text }}
			</div>
		</template>
	</a-table>

	<a-table v-bind="$attrs" :columns="FETAttackColumns" :dataSource="responseData.FET" :pagination="false" @expand="getInnerData">
		<template #title>
			<div class="flex justify-between pr-4">
				<h4>FET_Attack Results</h4>
			</div>
		</template>
		<template #bodyCell="{ column, text, record }">
			<div class="" v-if="column.dataIndex === 'index'">
				<div class="text-subtext">
					{{ record.at(-1)[0] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rouge1'">
				<div class="text-subtext">
					{{ record.at(-1)[1] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rouge2'">
				<div class="text-subtext">
					{{ record.at(-1)[2] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rougeL'">
				<div class="text-subtext">
					{{ record.at(-1)[3] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'wrr'">
				<div class="text-subtext">
					{{ record.at(-1)[4] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'distance'">
				<div class="text-subtext">
					{{ record.at(-1)[5] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'fr'">
				<div class="text-subtext">
					{{ record.at(-1)[6] }}
				</div>
			</div>
		</template>
		<template #expandedRowRender>
			<a-table :columns="FETInnoColumns" :data-source="FETInnerData" :pagination="false">
				<template #bodyCell="{ column, text, record }">
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
				</template>
			</a-table>
		</template>
	</a-table>
</template>