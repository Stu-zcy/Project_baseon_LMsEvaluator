import AttackTable from "./AttackTable.js";
import logViewer from "./logViewer.js";
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
		'AttackTable': AttackTable,
		'logViewer': logViewer
	},
	template: `
        <div>
            <div class="history-container">
                <div v-if="responseData && responseData.length > 0" class="records-list">
                        <div v-for="item in responseData" :key="item[0]" class="card shadow">
                            <div class="card-body">
																<!-- 0.名称（item[3]） -->
																<div class="record-name">
																	<h4>{{ item[3] }}</h4>
																</div>

                                <!-- 1. 时间 -->
                                <div class="record-time">
                                    {{ formatTime(item[0]) }}
                                </div>

                                <!-- 2. 状态图标 -->
                                <div class="status-container">                                  
																		<div v-if="(item[1] == 0)" class="executing" >
																				<button @click="showModal_log(item[0], item[1])" class="progress-button">
																				<a-progress :percent="computeProgress(item[4])" size="small" status="active"/>
																				</button>
																				<spin class="exe-spin">{{ getAttackType(item[4], item[5]) }}</spin>
																		</div>
                                    <div v-else-if="(item[1] == 2)" class="status-icon">
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
                                        <a-button type="primary" @click="showModal(item[0], item[1])">查看详情</a-button>
																				<a-Popconfirm title="确定删除此记录吗？" @confirm="del(item[0])" ok-text="确定" cancel-text="取消">
                                        <a-button danger>删除</a-button>
																				</a-Popconfirm>
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
                    title="执行详情"
                    width="90%"
                    wrap-class-name="full-modal-custom"
                    :footer="null"
                    :destroyOnClose="true"
                    @cancel="handleCancel"
                >
										<div class="report-buttons">
<a-button
      class="spawn-report-button"
      @click="generateReport"
      :disabled="spawnState === 1"
    >
      <span v-if="spawnState === 1" class="spawnstate">
        <a-spin :indicator="indicator" />
        <spin>正在生成</spin>
      </span>
      <span v-else-if="spawnState === 0" class="spawnstate">
				<img alt="spawn" src="../assets/icons/deepseek.svg" class="state-img"/>
				<spin>生成报告</spin>
			</span>
      <span v-else-if="spawnState === 2" class="spawnstate">
				<img alt="respawn" src="../assets/icons/deepseek.svg" class="state-img"/>
				<spin>重新生成</spin>
			</span>
    </a-button>

    <a-button
      class="read-report-button"
      :disabled="spawnState !== 2"
      @click="readReport"
    >
				<img alt="respawn" src="../assets/icons/read.svg" class="read-img"/>
				<spin>阅读报告</spin>
    </a-button>

    <a-button
      class="download-report-button"
      :disabled="spawnState !== 2"
      @click="downloadReport"
    >
				<img alt="respawn" src="../assets/icons/download.svg" class="download-img"/>
				<spin>下载报告</spin>
    </a-button>
										</div>
                    <AttackTable :target-create-time="targetCreateTime" />
                </a-modal>

								<a-modal
                    v-if="OPEN_log"
                    :open="OPEN_log"
                    title="执行日志"
                    width="90%"
                    wrap-class-name="full-modal-custom"
                    :footer="null"
                    :destroyOnClose="true"
                    @cancel="handleCancel_log"
                >
									<logViewer :username="targetUsername" :target-create-time="targetCreateTime" />
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
		const OPEN_log = ref(false);
		const targetUsername = ref(username);
		const targetCreateTime = ref(null);
		const starValues = ref({}); // Should be an object
		// Ensure LoadingOutlined is available, or provide a fallback
		const indicator = h('img', {
			src: '../assets/icons/loading.gif',
			alt: 'Loading',
			class: 'state-img'
		});
		const spawnState = ref(0);
		const reportID = ref("");

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
					console.log("Fetched records:", responseData.value);
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
			} finally {
				if ((responseData.value.length > 0)) {
					// let flag = false;
					for (let i = 0; i < responseData.value.length; ++i) {
						if (computeProgress(responseData.value[i][4]) !== '100.00') {
							setTimeout(fetchData, 5 * 1000);
							break;
						}
					}
				}
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

		async function showModal(createTimeVal, state) {
			if (state == 1) {
				targetCreateTime.value = createTimeVal;
				OPEN.value = true;
				console.log("展示对话框 (Showing modal for createTime):", createTimeVal);

				const response = await axios.post("/api/read_report", {
					username: username,
					createTime: targetCreateTime.value
				});
				console.log(response);
				spawnState.value = response.data.spawnState
				reportID.value = response.data.reportID

			} else if (state == 2) {
				// 执行失败
				antd.message.error("执行失败，无法查看详情。");
			} else {
				// 执行中
				antd.message.warning("当前实验正在执行中，请稍后再试。");
			}
		}

		function handleCancel() { // Changed from handleOK to handleCancel for clarity
			OPEN.value = false;
			spawnState.value = 0;
			reportID.value = "";
		}

		async function showModal_log(createTimeVal, state) {
			if (state == 0) {
				targetCreateTime.value = createTimeVal;
				OPEN_log.value = true;
				console.log("展示日志 (Showing modal for createTime):", createTimeVal);
			}
		}

		function handleCancel_log() {
			OPEN_log.value = false;
		}

		async function del(createTimeVal) {
			// Add confirmation dialog here if desired
			try {
				const response = await axios.post('/api/deleteRecord', {
					username: username,
					token: token,
					createTime: createTimeVal
				});
				antd.message.success("记录删除成功。");
				fetchData();
			} catch (error) {
				console.error("Error deleting record:", error);
				antd.message.error("删除记录失败，请稍后再试。");
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
		};

		async function generateReport() {
			console.log("即将生成...");
      antd.message.success("DeepSeek正在生成您的报告...");
			spawnState.value = 1; // 正在生成

      try {
        const response = await axios.post('/api/generate_report', {
          username: username,
					createTime: targetCreateTime.value,
        });

        if (response.status === 200) {
					spawnState.value = 2;
					reportID.value = response.data.reportID;
          antd.message.success('报告生成成功！');
        } else {
					spawnState.value = 0;
          antd.message.error('报告生成失败');
        }
      } catch (error) {
				spawnState.value = 0;
        console.error('生成报告失败:', error);
        antd.message.error('报告生成失败，请稍后再试。');
      }
    }

    function readReport() {
      if (spawnState.value === 2 && reportID.value !== "") {
        // 构建完整的URL，如果reportPath是相对路径，需要加上你的后端API基础URL
        const fullUrl = "/reports/" + reportID.value + ".pdf";
        // 使用 window.open 在新标签页打开PDF，浏览器会根据MIME类型处理
        window.open(fullUrl, '_blank');
      } else {
        antd.message.warn('请先生成报告！');
      }
    }

    function downloadReport() {
      if (spawnState.value === 2 && reportID.value !== "") {
				const fullUrl = "/reports/" + reportID.value + ".pdf";
				const link = document.createElement('a');
				link.href = fullUrl; // 同样直接使用
				link.download = fullUrl.substring(fullUrl.lastIndexOf('/') + 1);
				document.body.appendChild(link);
				link.click();
				document.body.removeChild(link);
      } else {
        antd.message.warn('请先生成报告！');
      }
    }

    // 
    function computeProgress(str) {
      const parts = str.split('/');
			console.log("总共", parts[1], "次攻击，正在执行第", parts[0], "次...");
			let progress = Number(parts[0]) / Number(parts[1]) * 100;
			progress = progress.toFixed(2);
			console.log("执行进度: ", progress);
      return progress;
    }

		// 获取正在进行的攻击名称
		function getAttackType(str, list) {
			const map = {
				"AdvAttack": "对抗样本攻击",
				"BackdoorAttack": "后门攻击",
				"PoisoningAttack": "数据投毒攻击",
				"FET": "梯度反演攻击",
				"RLMI": "模型反演攻击",
				"ModelStealingAttack": "模型窃取攻击"
			}
			const parts = str.split('/');
			if (list.length !== Number(parts[1])) {
				console.error("攻击数量不一致！");
				return "error";
			}
			if (Number(parts[0]) < Number(parts[1]))
				return map[list[Number(parts[0])]] + " 执行中...";
			else
				return "即将执行完毕..."
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
			OPEN_log,
			targetCreateTime,
			targetUsername,
			starValues,
			indicator,
			spawnState,
			reportID,
			fetchData,
			onPageChange,
			formatTime,
			showModal,
			handleCancel, // Expose the correct cancel handler
			showModal_log,
			handleCancel_log,
			del,
			treasure,
			generateReport,
			readReport,
			downloadReport,
			computeProgress,
			getAttackType,
		};
	}
});

export default AttackRecords;