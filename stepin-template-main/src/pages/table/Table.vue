<script lang="ts" setup>
import AuthorTable from './AuthorTable.vue';
import ProjectTable from './ProjectTable.vue';
import AttackTable from './AttackTable.vue';
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { Pagination, Modal, Button} from 'ant-design-vue';

const username = localStorage.getItem('Global_username');  // 从 localStorage 获取用户名
const token = localStorage.getItem('Global_token');
const responseData = ref();
const currentPage = ref(1);
const currentPageSize = ref(5);
const totalPagesNum = ref(1);
const OPEN = ref<boolean>(false);
const targetAttackID = ref<number>(null);

async function fetchData() {
	const response = await axios.post('http://localhost:5000/api/attackRecords', {
		username: username,
		token: token,
		currentPage: currentPage.value,
		currentPageSize: currentPageSize.value
	});
	responseData.value = response.data.records;
	console.log(response.data)
	totalPagesNum.value = response.data.pagination.totalPagesNum;
}

function onPageChange(page: number, pageSize: number) {
	fetchData();
}

function showModal(id: number) {
	targetAttackID.value = id;
	OPEN.value = true;
	console.log("展示对话框");
}

function handleOK(e: MouseEvent) {
	console.log(e);
	OPEN.value = false;
}

fetchData();
</script>
<template>
	<!-- <div class="table w-full">-->
	<!-- <AttackTable /> -->
	<!--</div> -->
	<div class="history">
		<div class="records">
			<div class="hcard" v-for="item, index in responseData">
				<div>{{ item.createTime }}</div>
				<div>{{ item.attackResult }}</div>
				<div><a-button type="primary" @click="showModal(item.attackID)">查看详情</a-button></div>
			</div>
		</div>
		<a-pagination v-model:current="currentPage" :total="totalPagesNum" v-model:pageSize="currentPageSize"
			:showSizeChanger="false" @change="onPageChange" style="text-align: center;"/>
		<a-modal v-if="OPEN" visible="true" title="hello" width = 80% wrap-class-name="full-modal" 
		:footer="null" destroyOnClose="true" @cancel="handleOK">
			<AttackTable :targetAttackID="targetAttackID"/>
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
	width: 80%;
	margin: 0 auto 5em;
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