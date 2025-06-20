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
        <div> <!-- Added a root div for good practice, though not strictly necessary if only one a-table per section -->
            <a-table v-bind="$attrs" :columns="AdvColumns" :dataSource="responseData.AdversarialAttack" :pagination="false">
                <template #title>
                    <div class="flex justify-between pr-4">
                        <h4>AdversarialAttack Results</h4>
                    </div>
                </template>
                <template #bodyCell="{ column, record }">
                    <div class="" v-if="column.dataIndex === 'index'">
                        <div class="text-subtext">
                            {{ record[0] && record[0][0] }}
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
                        {{ record[column.dataIndex] }} <!-- Fallback for other potential columns -->
                    </div>
                </template>
            </a-table>

            <a-table v-bind="$attrs" :columns="PoisoningColumns" :dataSource="responseData.PoisoningAttack" :pagination="false">
                <template #title>
                    <div class="flex justify-between pr-4">
                        <h4>PoisoningAttack Results</h4>
                    </div>
                </template>
                <template #bodyCell="{ column, record }">
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
                        {{ record[column.dataIndex] }}
                    </div>
                </template>
            </a-table>

            <a-table v-bind="$attrs" :columns="BackDoorColumns" :dataSource="responseData.BackDoorAttack" :pagination="false">
                <template #title>
                    <div class="flex justify-between pr-4">
                        <h4>BackDoorAttack Results</h4> <!-- Typo 'BackDoorAttackp' corrected -->
                    </div>
                </template>
                <template #bodyCell="{ column, record }">
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
                        {{ record[column.dataIndex] }}
                    </div>
                </template>
            </a-table>

            <a-table v-bind="$attrs" :columns="RLMIAttackColumns" :dataSource="responseData.RLMI" :pagination="false">
                <template #title>
                    <div class="flex justify-between pr-4">
                        <h4>RLMI_Attack Results</h4>
                    </div>
                </template>
                <template #bodyCell="{ column, record }">
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
                        {{ record[column.dataIndex] }}
                    </div>
                </template>
            </a-table>

            <a-table v-bind="$attrs" :columns="SWATAttackColumns" :dataSource="responseData.SWAT" :pagination="false" @expand="getInnerData">
                <template #title>
                    <div class="flex justify-between pr-4">
                        <h4>SWAT_Attack Results</h4>
                    </div>
                </template>
                <template #bodyCell="{ column, record }">
                    <div class="" v-if="column.dataIndex === 'index'">
                        <div class="text-subtext">
                            {{ record.at(-1) && record.at(-1)[0] }}
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
                    <a-table :columns="SWATInnoColumns" :data-source="SWATInnerData" :pagination="false">
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
        </div>
    `,

    setup(props) { // `props` is automatically passed here
        // Constants for columns
        const AdvColumns = [
            {
                title: '索引',
                dataIndex: 'index',
                // customRender: ({ text, index }) => { index } // This was unusual. Assuming you want to display the index+1
                customRender: ({ index }) => index + 1 // Corrected: typically display 1-based index
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
                 customRender: ({ index }) => index + 1 // Added for consistency if needed
            },
            { title: '准确率（攻击前）', dataIndex: 'before' },
            { title: '准确率（攻击后）', dataIndex: 'after' },
        ];
        const BackDoorColumns = [
            { title: '索引', dataIndex: 'index', customRender: ({ index }) => index + 1 },
            { title: '数据集', dataIndex: 'dataset' },
            { title: '投毒方', dataIndex: 'Poisoner' },
            { title: '准确率（原始数据集）', dataIndex: 'before' },
            { title: '准确率（毒化数据集）', dataIndex: 'after' },
            { title: 'PPL', dataIndex: 'PPL' },
            { title: 'USE', dataIndex: 'USE' },
            { title: 'GRAMMAR', dataIndex: 'GRAMMAR' },
        ];
        const RLMIAttackColumns = [
            { title: '索引', dataIndex: 'index', customRender: ({ index }) => index + 1 },
            { title: 'average ASR(Attack)', dataIndex: 'ASR_Attack' },
            { title: 'average WER(Attack)', dataIndex: 'WER_Attack' },
            { title: 'average ASR(inference)', dataIndex: 'ASR_Inference' },
            { title: 'average WER(inference)', dataIndex: 'WER_Inference' },
        ];
        const SWATAttackColumns = [
            { title: '轮', dataIndex: 'index' }, // This seems to refer to the outer loop index in template
            { title: 'average rouge1', dataIndex: 'rouge1' },
            { title: 'average rouge2', dataIndex: 'rouge2' },
            { title: 'average rougeL', dataIndex: 'rougeL' },
            { title: 'average Word recovery rate', dataIndex: 'wrr' },
            { title: 'average Edit distance', dataIndex: 'distance' },
            { title: 'full recovery rate', dataIndex: 'fr' },
        ];
        const SWATInnoColumns = [ // Typo "SWATInnoColumns" vs "SWATInnerColumns", using original
            { title: 'rouge1', dataIndex: 'rouge1' },
            { title: 'rouge2', dataIndex: 'rouge2' },
            { title: 'rougeL', dataIndex: 'rougeL' },
            { title: 'Word recovery rate', dataIndex: 'wrr' },
            { title: 'Edit distance', dataIndex: 'distance' },
            { title: 'if full recovery', dataIndex: 'fr' },
        ];

        const responseData = ref({
            AdversarialAttack: [],
            BackDoorAttack: [],
            PoisoningAttack: [],
            RLMI: [],
            SWAT: [],
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
                const response = await axios.post('http://127.0.0.1:5000/api/getRecord', {
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
                } else {
                    console.error("Invalid API response structure:", response);
                }
                console.log("Fetched responseData:", responseData.value);
            } catch (error) {
                console.error("请求出错 (Error fetching data):", error);
                // Reset data or show error message to user
                responseData.value = { AdversarialAttack: [], BackDoorAttack: [], PoisoningAttack: [], RLMI: [], SWAT: [] };
            }

            // The .map logic below to prepend an index seems to conflict with how
            // Ant Design Table's `customRender: ({index})` works for its own index.
            // If your data ALREADY has an index-like first element, then these maps might be okay,
            // but it makes the template access like `record[0][0]` more complex.
            // For simplicity with Ant Design Table, it's often better if `dataSource` items are flat objects
            // and `columns` `dataIndex` directly map to object keys.
            // However, I'll keep your original mapping logic. Be mindful of `record[0][0]` vs `record.index` etc.

            if (responseData.value.AdversarialAttack && Array.isArray(responseData.value.AdversarialAttack)) {
                // Your original map creates an array of arrays: [[index, ...item]]
                // This means in template: record[0] is the inner array, record[0][0] is index, record[0][1] is first data point etc.
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
            responseData,
            SWATInnerData,
            getInnerData,
            // username, token, props are available via `this` or closure, but not directly needed by template
        };
    }
});

export default AttackTable;