<script lang="ts" setup>
import AttackTable from "@/pages/table/AttackTable.vue"
import {ref, h} from "vue"
import axios from "axios"
import { LoadingOutlined } from "@ant-design/icons-vue";

const username = localStorage.getItem('Global_username');  // 从 localStorage 获取用户名
const token = localStorage.getItem('Global_token');
const responseData = ref([]);
const currentPage = ref(1);
const props = defineProps({
	defaultCurrentPageSize: Number,
	onlyTreasure: Boolean
})
const currentPageSize = ref(props.defaultCurrentPageSize)
const totalRecordsNum = ref(1);
const OPEN = ref<boolean>(false);
const targetCreateTime = ref<number>(null);
const starValues = ref({});
const indicator = h(LoadingOutlined, {
	style: {
    fontSize: '24px',
  },
  spin: true,
});


async function fetchData() {
	const response = await axios.post('http://127.0.0.1:5000/api/attackRecords', {
		username: username,
		token: token,
		currentPage: currentPage.value,
		currentPageSize: currentPageSize.value,
		onlyTreasure: props.onlyTreasure
	});
	responseData.value = response.data.records;
	totalRecordsNum.value = response.data.pagination.totalRecordsNum;
	responseData.value.forEach(item => {
		starValues.value[item[0]] = item[2]
	})
}

function onPageChange(page: number, pageSize: number) {
	fetchData();
}

function formatTime(createTime: number) {
	let time = new Date(createTime * 1000)
	return time.toLocaleString()
}

function showModal(createTime: number) {
	targetCreateTime.value = createTime
	OPEN.value = true;
	console.log("展示对话框");
}

function handleOK(e: MouseEvent) {
	console.log(e);
	OPEN.value = false;
}

async function del(createTime: number) {
	const response = await axios.post('http://127.0.0.1:5000/api/deleteRecord', {
		username: username,
		token: token,
		createTime: createTime
	});

	fetchData();
}

async function treasure(createTime: number, value: number) {
	starValues[createTime] = !starValues[createTime]
	const response = await axios.post('http://127.0.0.1:5000/api/treasure', {
		username: username,
		token: token,
		createTime: createTime,
		isTreasure: starValues[createTime]
	});

	fetchData();
}

fetchData();
</script>
<template>
	<div class="history w-4/5 mx-auto mt-4" v-if="responseData.length>0">
		<div class="records">
			<div class="hcard" v-for="item, index in responseData">
				<div>{{ formatTime(item[0]) }}</div>
				<div>
					<a-spin :indicator="indicator" v-if="item[1]==0"></a-spin>
					<div v-else-if="item[1]==2" class="w-8 h-8">
						<img src="@/assets/icons/fail.png" class="w-full h-full">
					</div>
					<div v-else class="w-8 h-8">
						<img src="@/assets/icons/complete.png" class="w-full h-full">
					</div>
				</div>
				<div style="display: flex; justify-content: space-between;width: 180px;">
					<a-rate 
					:count="1" :value="starValues[item[0]]" @change="(value: number) => {treasure(item[0], value)}" :allowClear="true"
					></a-rate>
					<div><a-button type="primary" @click="showModal(item[0])">查看详情</a-button></div>
					<div><a-button danger @click="del(item[0])">删除</a-button></div>
				</div>
			</div>
		</div>
		<a-pagination v-model:current="currentPage" :total="totalRecordsNum" v-model:pageSize="currentPageSize"
			:showSizeChanger="false" @change="onPageChange" style="text-align: center;"/>
		<a-modal v-if="OPEN" visible="true" title="hello" width = 80% wrap-class-name="full-modal" 
		:footer="null" destroyOnClose="true" @cancel="handleOK">
			<AttackTable :targetCreateTime="targetCreateTime"/>
		</a-modal>
	</div>


</template>
<style lang="less" scoped>
.records {
	min-height: 30em
}

.hcard {
	display: flex;
	border: 0px;
	border-radius: 25px;
	width: 100%;
	margin: 0 auto 50px;
	justify-content: space-between;
	align-items: center;
	padding: 50px;
	box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
}

.full-modal {
  .ant-modal {
    max-width: 100%;
    top: 0;
    padding-bottom: 0;
    margin: 0;
  }
  .ant-modal-content {
    display: flex;
    flex-direction: column;
    height: calc(100vh);
  }
  .ant-modal-body {
    flex: 1;
  }
}
</style>