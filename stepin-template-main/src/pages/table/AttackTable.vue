<script lang="ts" setup>
import { ref, onMounted, customRef } from 'vue';
import axios from "axios"
import { number } from 'echarts';

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
const SWATAttackColumns = [
	{
		title: '索引',
		dataIndex: 'index',
	},
	{ title: 'Reference', dataIndex: 'refer' },
	{ title: 'prediction', dataIndex: 'pred' },
	{ title: 'rouge1', dataIndex: 'rouge1' },
	{ title: 'rouge2', dataIndex: 'rouge2' },
	{ title: 'rouge3', dataIndex: 'rouge3' },
	{ title: 'Word recovery rate', dataIndex: 'rate' },
	{ title: 'Edit distance', dataIndex: 'distance' },
	{ title: 'If full recovery', dataIndex: 'ifFull' },
];

const responseData = ref({
	AdvAttack: [],
	BackDoorAttack: [],
	PoisoningAttack: [],
	SWAT: []
});

const BackDoorAttack = ref([])
// const SWATAttack = ref([])

const username = localStorage.getItem('Global_username');  // 从 localStorage 获取用户名
const token = localStorage.getItem('Global_token');
const props = defineProps({
	targetAttackID: Number
})
onMounted(async () => {
	try {
		const response = await axios.post('http://localhost:5000/api/getRecord', {
			username: username,
			token: token,
			attackID: props.targetAttackID
		});
		responseData.value = response.data;
		console.log(responseData.value);
	} catch (error) {
		console.error("请求出错", error);
	}

	if (responseData.value.AdvAttack) {
		responseData.value.AdvAttack = responseData.value.AdvAttack.map((item, index) => {
			return [[index + 1, ...(item[0])]].concat(item.slice(1));
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
	if (responseData.value.SWAT) {
		responseData.value.SWAT.forEach((item, index) => {
			let count = 0
			item = item.map((element) => {
				if (Array.isArray(element)) {
					return [++count, ...element]
				} else {
					return element
				}
			})
		})
	}

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
	<a-table v-bind="$attrs" :columns="AdvColumns" :dataSource="responseData.AdvAttack" :pagination="false">
		<template #title>
			<div class="flex justify-between pr-4">
				<h4>AdvAttack Results</h4>
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

	<!-- <a-table v-for="(items, index) in responseData.SWAT" v-bind="$attrs" :columns="SWATAttackColumns" :dataSource="items.filter((item) => Array.isArray(item))" :pagination="false">
		<template #title>
			<div class="flex justify-between pr-4">
				<h4>SWAT_Attack Results</h4>
			</div>
		</template>
		<template #bodyCell="{ column, text, record }">
			<div class="" v-if="column.dataIndex === 'index'">
				<div class="text-subtext">
					{{ record[0] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'refer'">
				<div class="text-subtext">
					{{ record[1] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'pred'">
				<div class="text-subtext">
					{{ record[2] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rouge1'">
				<div class="text-subtext">
					{{ record[3] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rouge2'">
				<div class="text-subtext">
					{{ record[4] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rouge3'">
				<div class="text-subtext">
					{{ record[5] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rate'">
				<div class="text-subtext">
					{{ record[6] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'distance'">
				<div class="text-subtext">
					{{ record[7] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'ifFull'">
				<div class="text-subtext">
					{{ record[8] }}
				</div>
			</div>
			<div v-else class="text-subtext">
				{{ text }}
			</div>
		</template>
	</a-table> -->
</template>