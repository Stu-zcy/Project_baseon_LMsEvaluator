import AttackTable from "./AttackTable.js";
import axios from "../utils/axiosConfig.js";

const { ref, h, defineComponent, onMounted } = Vue;

// 全局分页器样式的处理（这是唯一需要 JS 注入的部分，因为它修改的是第三方组件 antd）
const darkPaginationStyle = `
  .ant-pagination-item, .ant-pagination-item-link { background-color: rgba(255, 255, 255, 0.1) !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; backdrop-filter: blur(4px) !important; -webkit-backdrop-filter: blur(4px) !important; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4); }
  .ant-pagination-item a { color: rgba(255, 255, 255, 0.85) !important; }
  .ant-pagination-item-active { background-color: rgba(255, 255, 255, 0.3) !important; border-color: rgba(255, 255, 255, 0.5) !important; font-weight: 600; }
  .ant-pagination-item-active a { color: #fff !important; }
  .ant-pagination-item:not(.ant-pagination-item-active):hover, .ant-pagination-item-link:not(.ant-pagination-disabled):hover { background-color: rgba(255, 255, 255, 0.2) !important; border-color: rgba(255, 255, 255, 0.4) !important; }
  .ant-pagination-item-link { color: rgba(255, 255, 255, 0.85) !important;}
  .ant-pagination-disabled .ant-pagination-item-link, .ant-pagination-disabled:hover .ant-pagination-item-link { color: rgba(255, 255, 255, 0.3) !important; background-color: rgba(255, 255, 255, 0.05) !important; border-color: rgba(255, 255, 255, 0.1) !important; backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }
  .ant-pagination-jump-prev .ant-pagination-item-container .ant-pagination-item-ellipsis, .ant-pagination-jump-next .ant-pagination-item-container .ant-pagination-item-ellipsis { color: rgba(255, 255, 255, 0.85) !important; }
	.empty-message {text-align: center;	padding: 3rem 0;color: #fff;}
`;

const lightPaginationStyle = `
  /* Light Theme Pagination Styles (Ant Design Defaults) */
  .ant-pagination-item a { color: rgba(0, 0, 0, 0.85) !important; }
  .ant-pagination-item-active { background-color: #1890ff !important; border-color: #1890ff !important; }
  .ant-pagination-item-active a { color: #fff !important; }
  .ant-pagination-item-link { color: rgba(0, 0, 0, 0.85) !important; background-color: #fff !important; }
  .ant-pagination-disabled .ant-pagination-item-link, .ant-pagination-disabled a { color: rgba(0, 0, 0, 0.25) !important; }
  .ant-pagination-jump-prev .ant-pagination-item-container .ant-pagination-item-ellipsis,
  .ant-pagination-jump-next .ant-pagination-item-container .ant-pagination-item-ellipsis { color: rgba(0, 0, 0, 0.85) !important; }
	.empty-message {text-align: center;	padding: 3rem 0;color: #344767;} /* 黑色用black，深蓝色用#344767 */
`;

function injectGlobalStyle(css, id) {
	if (document.getElementById(id)) return;
	const styleElement = document.createElement('style');
	styleElement.id = id;
	styleElement.appendChild(document.createTextNode(css));
	document.head.appendChild(styleElement);
}

const AttackRecords = defineComponent({
	name: 'AttackRecords',
	props: {
		defaultCurrentPageSize: {
			type: Number,
			default: 5 // Example default
		},
		onlyTreasure: {
			type: Boolean,
			default: false
		},
		theme: {
			type: String,
			default: 'dark', // 默认是暗色主题
			validator: (value) => ['light', 'dark'].includes(value) // 验证传入的值是否合法
		}
	},
	components: {
		'AttackTable': AttackTable
	},
	template: `
        <div>
            <div class="history-container">
                <div v-if="responseData && responseData.length > 0">
                    <div class="records-list">
                        <div v-for="item in responseData" :key="item[0]" class="card shadow">
                            <div class="card-body">
                                <!-- 1. 时间 -->
                                <div class="record-time">
                                    {{ formatTime(item[0]) }}
                                </div>

                                <!-- 2. 状态图标 -->
                                <div class="status-container">
                                    <a-spin :indicator="indicator" v-if="item[1] === 0"></a-spin>
                                    <div v-else-if="item[1] === 2" class="status-icon">
                                        <img src="../assets/icons/fail.png" alt="Fail">
                                    </div>
                                    <div v-else class="status-icon">
                                        <img src="../assets/icons/complete.png" alt="Complete">
                                    </div>
                                </div>

                                <!-- 3. 操作按钮 -->
                                <div class="actions-container">
                                    <div class="actions-bar">
                                        <a-rate
                                            :count="1"
                                            :value="starValues[item[0]] ? 1 : 0"
                                            @change="(value) => treasure(item[0], value)"
                                            :allowClear="false"
                                        ></a-rate>
                                        <a-button type="primary" @click="showModal(item[0])">查看详情</a-button>
                                        <a-button danger @click="del(item[0])">删除</a-button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div v-else class="empty-message">
                    <p>暂无执行记录。</p>
                </div>

                <!-- 分页器 -->
                <a-pagination
                    v-if="responseData && responseData.length > 0"
                    v-model:current="currentPage"
                    :total="totalRecordsNum"
                    v-model:pageSize="currentPageSize"
                    :showSizeChanger="false"
                    @change="onPageChange"
                    class="pagination-container"
                />

                <!-- Modal -->
                <a-modal
                    v-if="OPEN"
                    :open="OPEN"
                    title="攻击详情"
                    width="80%"
                    wrap-class-name="full-modal-custom"
                    :footer="null"
                    :destroyOnClose="true"
                    @cancel="handleCancel"
                >
                    <AttackTable :target-create-time="targetCreateTime" />
                </a-modal>
            </div>
        </div>
    `,
	setup(props) {
		const username = localStorage.getItem('Global_username');
		const token = localStorage.getItem('Global_token');
		const responseData = ref([]);
		const currentPage = ref(1);
		const currentPageSize = ref(props.defaultCurrentPageSize);
		const totalRecordsNum = ref(1);
		const OPEN = ref(false);
		const targetCreateTime = ref(null);
		const starValues = ref({}); // Should be an object

		// Ensure LoadingOutlined is available, or provide a fallback
		const indicator = h(window.LoadingOutlined || 'div', { // Use window.LoadingOutlined
			style: {
				fontSize: '24px',
			},
			spin: true,
		});

		async function fetchData() {
			// Ensure axios is available globally
			if (typeof axios === 'undefined') {
				console.error("axios is not defined. Make sure it's loaded globally via CDN.");
				return;
			}
			try {
				
				const response = await axios.post('/api/attackRecords', {
					username: username,
					token: token,
					currentPage: currentPage.value,
					currentPageSize: currentPageSize.value,
					onlyTreasure: props.onlyTreasure
				});
				if (response.data && response.data.records) {
					responseData.value = response.data.records;
					totalRecordsNum.value = response.data.pagination.totalRecordsNum;
					const newStarValues = {};
					responseData.value.forEach(item => {
						// item[0] is createTime, item[2] is isTreasure (boolean or 0/1)
						newStarValues[item[0]] = !!item[2]; // Ensure boolean for consistency
					});
					starValues.value = newStarValues;
				} else {
					responseData.value = [];
					totalRecordsNum.value = 1;
					starValues.value = {};
				}
			} catch (error) {
				console.error("Error fetching records:", error);
				responseData.value = [];
				totalRecordsNum.value = 1;
				starValues.value = {};
			}
		}

		function onPageChange(page, pageSize) { // pageSize might not be used if showSizeChanger is false
			// currentPage.value is already updated by v-model
			// currentPageSize.value is already updated by v-model (if changer enabled)
			fetchData();
		}

		function formatTime(createTime) {
			if (!createTime) return '';
			let time = new Date(createTime * 1000);
			return time.toLocaleString();
		}

		function showModal(createTimeVal) {
			targetCreateTime.value = createTimeVal;
			OPEN.value = true;
			console.log("展示对话框 (Showing modal for createTime):", createTimeVal);
		}

		function handleCancel() { // Changed from handleOK to handleCancel for clarity
			OPEN.value = false;
		}

		async function del(createTimeVal) {
			// Add confirmation dialog here if desired
			try {
				const response = await axios.post('/api/deleteRecord', {
					username: username,
					token: token,
					createTime: createTimeVal
				});
				// Assuming successful deletion, re-fetch data
				fetchData();
			} catch (error) {
				console.error("Error deleting record:", error);
				// Optionally show an error message to the user
			}
		}

		async function treasure(createTimeVal, /* newRateValue */) {
			// a-rate with count=1 and :value="starValues[item[0]] ? 1 : 0"
			// The 'change' event for a-rate with count=1 will give a value of 0 if unstarred, 1 if starred.
			// So we can just toggle the existing boolean state.
			const currentIsTreasure = starValues.value[createTimeVal];
			const newIsTreasure = !currentIsTreasure;

			try {
				const response = await axios.post('/api/treasure', {
					username: username,
					token: token,
					createTime: createTimeVal,
					isTreasure: newIsTreasure
				});
				// Optimistically update UI, or re-fetch for consistency
				starValues.value[createTimeVal] = newIsTreasure;
				// If API response confirms or if you prefer single source of truth, call fetchData()
				// fetchData(); // Uncomment if you prefer to refetch all data
			} catch (error) {
				console.error("Error treasuring record:", error);
				// Optionally revert UI or show error
			}
		}

		onMounted(() => {
			// 只注入分页器的样式
			const styleToInject = props.theme === 'light' ? lightPaginationStyle : darkPaginationStyle;
			const styleId = `pagination-style-${props.theme}`; // 使用动态ID以防万一
			
			injectGlobalStyle(styleToInject, styleId);
		});

		// setup中的内容会运行一次
		fetchData(); // Initial data fetch

		return {
			responseData,
			currentPage,
			currentPageSize,
			totalRecordsNum,
			OPEN,
			targetCreateTime,
			starValues,
			indicator,
			fetchData,
			onPageChange,
			formatTime,
			showModal,
			handleCancel, // Expose the correct cancel handler
			del,
			treasure
		};
	}
});

export default AttackRecords;