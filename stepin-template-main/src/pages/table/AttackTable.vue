<script lang="ts" setup>
	import { getBase64 } from '@/utils/file';
  import { FormInstance } from 'ant-design-vue';
  import { reactive, ref, onMounted, customRef } from 'vue';
  import dayjs from 'dayjs';
  import { Dayjs } from 'dayjs';
  import { EditFilled } from '@ant-design/icons-vue';
	import axios from "axios"
import { title } from 'process';
import { rowProps } from 'ant-design-vue/lib/grid/Row';
import { Item } from 'ant-design-vue/lib/menu';

	const AdvColumns = [
    {
      title: '索引',
      dataIndex: 'index',
			customRender: ({text,index }) => {index}
    },
    { title: '成功攻击次数', dataIndex: 'success' },
    { title: '失败攻击次数', dataIndex: 'fail' },
    { title: '跳过攻击次数', dataIndex: 'skip' },
    { title: '准确率（攻击前）', dataIndex: 'before'},
		{ title: '准确率（攻击后）', dataIndex: 'after'},
		{ title: '成功率', dataIndex: 'rate'},
  ];
	const PoisoningColumns = [
    {
      title: '索引',
      dataIndex: 'index',
    },
    { title: '准确率（攻击前）', dataIndex: 'before'},
		{ title: '准确率（攻击后）', dataIndex: 'after'},
	];
	let BackDoorColumns = [];
	const SWATAttackColumns = [
    {
      title: '索引',
      dataIndex: 'index',
    },
    { title: 'Reference', dataIndex: 'refer'},
    { title: 'prediction', dataIndex: 'pred'},
    { title: 'rouge1', dataIndex: 'rouge1'},
    { title: 'rouge2', dataIndex: 'rouge2'},
    { title: 'rouge3', dataIndex: 'rouge3'},
    { title: 'Word recovery rate', dataIndex: 'rate'},
    { title: 'Edit distance', dataIndex: 'distance'},
    { title: 'If full recovery', dataIndex: 'ifFull'},
	];

	const responseData = ref({
		AdvAttack: [],
		BackDoorAttack: [],
		PoisoningAttack: [],
		SWAT: []
	});

	const BackDoorAttack = ref([])
	// const SWATAttack = ref([])
	onMounted(async () => {
		try {
			const response = await axios.post('http://localhost:5000/api/submit_log');
			responseData.value = JSON.parse(response.data);
			console.log(responseData.value);
		} catch (error) {
			console.error("请求出错", error);
		}

		responseData.value.AdvAttack = responseData.value.AdvAttack.map((item, index) => {
			return [index + 1, ...item];
		});
		responseData.value.PoisoningAttack = responseData.value.PoisoningAttack.map((item, index) => {
			return [index + 1, ...item];
		});
		responseData.value.BackDoorAttack.forEach((item, index) => {
			item.forEach((element: Array<string>) => {
				BackDoorAttack.value.push([index + 1, ...element])
			});
		})
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
		BackDoorColumns = [
		{ title: '索引', 
			dataIndex: 'index', 
			customCell: ((record, rowIndex) => {
				if (record[1] != 'epoch:0')
					return {rowSpan: 0}
				// else 
				// 	return {rowSpan: 4}
				let count = 0
				while (BackDoorAttack.value[rowIndex+count][1] != '/' && rowIndex+count < BackDoorAttack.value.length)
					count++;
				console.log(count)
				if (rowIndex+count < BackDoorAttack.value.length)
					return {rowSpan: count+1}
				else {
					console.error("缺少'/', 请检查数据是否完整'")
					return
				}
			})
		},
		{ title: 'Epoach', dataIndex: 'sub_index'},
		{ title: '损失函数', dataIndex: 'loss'},
		{ title: '准确率（原数据集）', dataIndex: 'clean'},
		{ title: '准确率（污染数据集）', dataIndex: 'poison'},
	]
		console.log(BackDoorAttack.value)
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
					{{ record[0] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'success'">
				<div class="text-subtext">
					{{ record[1] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'fail'">
				<div class="text-subtext">
					{{ record[2] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'skip'">
				<div class="text-subtext">
					{{ record[3] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'before'">
				<div class="text-subtext">
					{{ record[4] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'after'">
				<div class="text-subtext">
					{{ record[5] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'rate'">
				<div class="text-subtext">
					{{ record[6] }}
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
	<a-table v-bind="$attrs" :columns="BackDoorColumns" :dataSource="BackDoorAttack" :pagination="false">
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
			<div class="" v-else-if="column.dataIndex === 'sub_index'">
				<div class="text-subtext">
					{{ record[1] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'loss'">
				<div class="text-subtext">
					{{ record[2] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'clean'">
				<div class="text-subtext">
					{{ record[3] }}
				</div>
			</div>
			<div class="" v-else-if="column.dataIndex === 'poison'">
				<div class="text-subtext">
					{{ record[4] }}
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