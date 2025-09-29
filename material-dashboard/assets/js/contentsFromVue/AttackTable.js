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
<a-table v-if="responseData.NormalTrain.length > 0" v-bind="$attrs" 
        :columns="NormalTrainColumns" :dataSource="responseData.NormalTrain" 
        :pagination="false" class="outer-table normal-train-table">
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
                <h4>常规训练</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
                    常规训练
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'acc'">
                <div class="text-subtext">
                    {{ convert(record[0], 2, true) }}
                </div>
            </div>
            <div class="" v-else-if="column.dataIndex === 'f1'">
                <div class="text-subtext">
                    {{ convert(record[1], 2) }}
                </div>
            </div>
            <div v-else class="text-subtext">
                {{ record[column.dataIndex] }}
            </div>
        </template>
    </a-table>


    <a-table v-if="responseData.AdvAttack.length > 0" v-bind="$attrs" 
        :columns="AdvColumns" :dataSource="responseData.AdvAttack" 
        :pagination="false" class="outer-table safe-attack-table" :rowClassName="rowClassName">
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
<svg v-if="record.info.defenderEnabled" style="font-size: 12px;" t="1753282514052" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="26441" width="200" height="200"><path d="M260.019513 846.71367C122.197171 732.496698 4.0592 531.652621 4.0592 216.591573A781.279013 781.279013 0 0 1 27.696319 11.810622a1210.477661 1210.477661 0 0 0 252.023439 35.447742A687.889375 687.889375 0 0 0 508.106079 0a725.464298 725.464298 0 0 0 232.339069 47.258364c110.264223 0 252.039314-35.447741 252.039314-35.447742A916.196323 916.196323 0 0 1 1020.010832 212.718197c0 319.077295-102.374601 515.921-240.212817 630.122097-82.706105 66.942734-271.723685 181.159706-271.723685 181.159706s-173.238334-110.327721-248.054817-177.28633z m39.368741-59.08486a2289.371655 2289.371655 0 0 0 208.543206 149.553592V508.174247h395.973335a1053.875794 1053.875794 0 0 0 37.38443-291.5668 816.060025 816.060025 0 0 0-7.937246-114.216972 998.616687 998.616687 0 0 1-196.843705 23.637119A757.340278 757.340278 0 0 1 508.106079 82.706105h-0.174619v425.436393h-390.512511a603.421201 603.421201 0 0 0 181.969305 279.502186z" fill="#1C87FF" p-id="26442">
</path>
</svg>
								<svg v-else style="font-size: 12px;" t="1753281965668" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="24625" width="200" height="200">
								<path d="M771.1744 208.4352l-44.5952 139.3152-150.4256 122.5728-33.4336 27.8528 27.8528 33.4336 83.5584 105.8816 33.4336 39.0144 33.4336-33.4336 44.5952-44.5952 16.6912 16.6912-50.1248 44.5952-33.4336 33.4336 33.4336 33.4336 78.0288 77.9776 5.5808 5.5808-16.6912 16.6912-5.5808-5.5808-78.0288-77.9776-33.4336-33.4336-27.8528 27.8528-44.5952 44.5952-16.6912-16.6912 44.5952-44.5952 33.4336-33.4336-39.0144-27.8528-116.992-94.72-27.8528-22.272-27.8528 22.272-116.992 94.72-39.0144 33.4336 33.4336 33.4336 44.5952 44.5952-16.6912 16.6912-44.5952-50.1248-33.4336-33.4336-33.4336 33.4336-78.0288 78.0288-5.5808 5.5808-16.6912-16.6912 5.5808-5.5808 78.0288-78.0288 33.4336-33.4336-27.8528-27.8528-44.5952-50.1248 16.6912-16.6912 44.5952 44.5952 33.4336 33.4336 27.8528-33.4336 83.5584-100.3008 27.8528-39.0144-33.4336-27.8528-150.4256-122.5728-44.5952-139.3152 139.3152 44.5952 111.36 133.7344 33.4336 44.5952 33.4336-44.5952 111.4112-133.7344 139.3152-44.5952m72.448-72.5504L609.5872 214.016 487.0144 358.8608 369.8688 219.5456 130.4576 135.9872l78.0288 234.0352 156.0064 128-78.0288 111.5648-72.448-78.0288-83.5584 78.0288 78.0288 78.0288-78.0288 78.1312-39.0144-39.0144-39.0144 39.0144 156.0064 156.0064 39.0144-39.0144-39.0144-39.0144 78.0288-77.9776 78.0288 78.0288 77.9776-78.0288-72.6016-72.6016 116.992-94.72 116.992 94.72-72.2432 72.6016 78.0288 78.0288 77.9776-78.0288 78.0288 78.0288-39.0144 39.0144 39.0144 39.0144 156.0064-156.0576-39.0144-39.0144-39.0144 39.0144-77.9776-78.1312 78.0288-78.0288-78.0288-77.9776-72.448 72.448-83.5584-100.3008 156.0064-128z" fill="#c83939ff" p-id="24626">
								</path>
								</svg>
								{{ record.info.name }}
                </div>
            </div>
<div class="" v-if="column.dataIndex === 'success'">
      <div class="text-subtext" v-if="record.compResult && record.compResult[0]" style="color: blue;">
        {{ convert(record.resultData[0], 2) }}
      </div>
      <div class="text-subtext" v-else>
        {{ convert(record.resultData[0], 2) }}
      </div>
    </div>
    <div class="" v-else-if="column.dataIndex === 'fail'">
      <div class="text-subtext" v-if="record.compResult && record.compResult[1]" style="color: blue;">
        {{ convert(record.resultData[1], 2) }}
      </div>
      <div class="text-subtext" v-else>
        {{ convert(record.resultData[1], 2) }}
      </div>
    </div>
    <div class="" v-else-if="column.dataIndex === 'skip'">
      <div class="text-subtext" v-if="record.compResult && record.compResult[2]" style="color: blue;">
        {{ convert(record.resultData[2], 2) }}
      </div>
      <div class="text-subtext" v-else>
        {{ convert(record.resultData[2], 2) }}
      </div>
    </div>
    <div class="" v-else-if="column.dataIndex === 'before'">
      <div class="text-subtext" v-if="record.compResult && record.compResult[3]" style="color: blue;">
        {{ convert(record.resultData[3], 2, true) }}
      </div>
      <div class="text-subtext" v-else>
        {{ convert(record.resultData[3], 2, true) }}
      </div>
    </div>
    <div class="" v-else-if="column.dataIndex === 'after'">
      <div class="text-subtext" v-if="record.compResult && record.compResult[4]" style="color: blue;">
        {{ convert(record.resultData[4], 2, true) }}
      </div>
      <div class="text-subtext" v-else>
        {{ convert(record.resultData[4], 2, true) }}
      </div>
    </div>
    <div class="" v-else-if="column.dataIndex === 'rate'">
      <div class="text-subtext" v-if="record.compResult && record.compResult[5]" style="color: blue;">
        {{ convert(record.resultData[5], 2, true) }}
      </div>
      <div class="text-subtext" v-else>
        {{ convert(record.resultData[5], 2, true) }}
      </div>
		</div>
        </template>
    </a-table>

    <a-table v-if="responseData.PoisoningAttack.length > 0" v-bind="$attrs" 
        :columns="PoisoningColumns" :dataSource="responseData.PoisoningAttack" 
        :pagination="false" class="outer-table safe-attack-table" :rowClassName="rowClassName">
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
                <h4>投毒攻击</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
<svg v-if="record.info.defenderEnabled" style="font-size: 12px;" t="1753282514052" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="26441" width="200" height="200"><path d="M260.019513 846.71367C122.197171 732.496698 4.0592 531.652621 4.0592 216.591573A781.279013 781.279013 0 0 1 27.696319 11.810622a1210.477661 1210.477661 0 0 0 252.023439 35.447742A687.889375 687.889375 0 0 0 508.106079 0a725.464298 725.464298 0 0 0 232.339069 47.258364c110.264223 0 252.039314-35.447741 252.039314-35.447742A916.196323 916.196323 0 0 1 1020.010832 212.718197c0 319.077295-102.374601 515.921-240.212817 630.122097-82.706105 66.942734-271.723685 181.159706-271.723685 181.159706s-173.238334-110.327721-248.054817-177.28633z m39.368741-59.08486a2289.371655 2289.371655 0 0 0 208.543206 149.553592V508.174247h395.973335a1053.875794 1053.875794 0 0 0 37.38443-291.5668 816.060025 816.060025 0 0 0-7.937246-114.216972 998.616687 998.616687 0 0 1-196.843705 23.637119A757.340278 757.340278 0 0 1 508.106079 82.706105h-0.174619v425.436393h-390.512511a603.421201 603.421201 0 0 0 181.969305 279.502186z" fill="#1C87FF" p-id="26442">
</path>
</svg>
								<svg v-else style="font-size: 12px;" t="1753281965668" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="24625" width="200" height="200">
								<path d="M771.1744 208.4352l-44.5952 139.3152-150.4256 122.5728-33.4336 27.8528 27.8528 33.4336 83.5584 105.8816 33.4336 39.0144 33.4336-33.4336 44.5952-44.5952 16.6912 16.6912-50.1248 44.5952-33.4336 33.4336 33.4336 33.4336 78.0288 77.9776 5.5808 5.5808-16.6912 16.6912-5.5808-5.5808-78.0288-77.9776-33.4336-33.4336-27.8528 27.8528-44.5952 44.5952-16.6912-16.6912 44.5952-44.5952 33.4336-33.4336-39.0144-27.8528-116.992-94.72-27.8528-22.272-27.8528 22.272-116.992 94.72-39.0144 33.4336 33.4336 33.4336 44.5952 44.5952-16.6912 16.6912-44.5952-50.1248-33.4336-33.4336-33.4336 33.4336-78.0288 78.0288-5.5808 5.5808-16.6912-16.6912 5.5808-5.5808 78.0288-78.0288 33.4336-33.4336-27.8528-27.8528-44.5952-50.1248 16.6912-16.6912 44.5952 44.5952 33.4336 33.4336 27.8528-33.4336 83.5584-100.3008 27.8528-39.0144-33.4336-27.8528-150.4256-122.5728-44.5952-139.3152 139.3152 44.5952 111.36 133.7344 33.4336 44.5952 33.4336-44.5952 111.4112-133.7344 139.3152-44.5952m72.448-72.5504L609.5872 214.016 487.0144 358.8608 369.8688 219.5456 130.4576 135.9872l78.0288 234.0352 156.0064 128-78.0288 111.5648-72.448-78.0288-83.5584 78.0288 78.0288 78.0288-78.0288 78.1312-39.0144-39.0144-39.0144 39.0144 156.0064 156.0064 39.0144-39.0144-39.0144-39.0144 78.0288-77.9776 78.0288 78.0288 77.9776-78.0288-72.6016-72.6016 116.992-94.72 116.992 94.72-72.2432 72.6016 78.0288 78.0288 77.9776-78.0288 78.0288 78.0288-39.0144 39.0144 39.0144 39.0144 156.0064-156.0576-39.0144-39.0144-39.0144 39.0144-77.9776-78.1312 78.0288-78.0288-78.0288-77.9776-72.448 72.448-83.5584-100.3008 156.0064-128z" fill="#c83939ff" p-id="24626">
								</path>
								</svg>
								{{ record.info.name }}
                </div>
            </div>
<div v-else-if="column.dataIndex === 'acc'">
      <div class="text-subtext" v-if="record.compResult && record.compResult[0]" style="color: blue;">
        {{ convert(record.resultData[0], 2, true) }}
      </div>
      <div class="text-subtext" v-else>
        {{ convert(record.resultData[0], 2, true) }}
      </div>
    </div>
    <div v-else-if="column.dataIndex === 'f1'">
      <div class="text-subtext" v-if="record.compResult && record.compResult[1]" style="color: blue;">
        {{ convert(record.resultData[1], 2) }}
      </div>
      <div class="text-subtext" v-else>
        {{ convert(record.resultData[1], 2) }}
      </div>
    </div>
            <div v-else class="text-subtext">
                {{ record.resultData[column.dataIndex] }}
            </div>
        </template>
    </a-table>

    <a-table v-if="responseData.BackdoorAttack.length > 0" v-bind="$attrs" 
        :columns="BackDoorColumns" :dataSource="responseData.BackdoorAttack" 
        :pagination="false" class="outer-table safe-attack-table" :rowClassName="rowClassName">
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
<svg v-if="record.info.defenderEnabled" style="font-size: 12px;" t="1753282514052" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="26441" width="200" height="200"><path d="M260.019513 846.71367C122.197171 732.496698 4.0592 531.652621 4.0592 216.591573A781.279013 781.279013 0 0 1 27.696319 11.810622a1210.477661 1210.477661 0 0 0 252.023439 35.447742A687.889375 687.889375 0 0 0 508.106079 0a725.464298 725.464298 0 0 0 232.339069 47.258364c110.264223 0 252.039314-35.447741 252.039314-35.447742A916.196323 916.196323 0 0 1 1020.010832 212.718197c0 319.077295-102.374601 515.921-240.212817 630.122097-82.706105 66.942734-271.723685 181.159706-271.723685 181.159706s-173.238334-110.327721-248.054817-177.28633z m39.368741-59.08486a2289.371655 2289.371655 0 0 0 208.543206 149.553592V508.174247h395.973335a1053.875794 1053.875794 0 0 0 37.38443-291.5668 816.060025 816.060025 0 0 0-7.937246-114.216972 998.616687 998.616687 0 0 1-196.843705 23.637119A757.340278 757.340278 0 0 1 508.106079 82.706105h-0.174619v425.436393h-390.512511a603.421201 603.421201 0 0 0 181.969305 279.502186z" fill="#1C87FF" p-id="26442">
</path>
</svg>
								<svg v-else style="font-size: 12px;" t="1753281965668" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="24625" width="200" height="200">
								<path d="M771.1744 208.4352l-44.5952 139.3152-150.4256 122.5728-33.4336 27.8528 27.8528 33.4336 83.5584 105.8816 33.4336 39.0144 33.4336-33.4336 44.5952-44.5952 16.6912 16.6912-50.1248 44.5952-33.4336 33.4336 33.4336 33.4336 78.0288 77.9776 5.5808 5.5808-16.6912 16.6912-5.5808-5.5808-78.0288-77.9776-33.4336-33.4336-27.8528 27.8528-44.5952 44.5952-16.6912-16.6912 44.5952-44.5952 33.4336-33.4336-39.0144-27.8528-116.992-94.72-27.8528-22.272-27.8528 22.272-116.992 94.72-39.0144 33.4336 33.4336 33.4336 44.5952 44.5952-16.6912 16.6912-44.5952-50.1248-33.4336-33.4336-33.4336 33.4336-78.0288 78.0288-5.5808 5.5808-16.6912-16.6912 5.5808-5.5808 78.0288-78.0288 33.4336-33.4336-27.8528-27.8528-44.5952-50.1248 16.6912-16.6912 44.5952 44.5952 33.4336 33.4336 27.8528-33.4336 83.5584-100.3008 27.8528-39.0144-33.4336-27.8528-150.4256-122.5728-44.5952-139.3152 139.3152 44.5952 111.36 133.7344 33.4336 44.5952 33.4336-44.5952 111.4112-133.7344 139.3152-44.5952m72.448-72.5504L609.5872 214.016 487.0144 358.8608 369.8688 219.5456 130.4576 135.9872l78.0288 234.0352 156.0064 128-78.0288 111.5648-72.448-78.0288-83.5584 78.0288 78.0288 78.0288-78.0288 78.1312-39.0144-39.0144-39.0144 39.0144 156.0064 156.0064 39.0144-39.0144-39.0144-39.0144 78.0288-77.9776 78.0288 78.0288 77.9776-78.0288-72.6016-72.6016 116.992-94.72 116.992 94.72-72.2432 72.6016 78.0288 78.0288 77.9776-78.0288 78.0288 78.0288-39.0144 39.0144 39.0144 39.0144 156.0064-156.0576-39.0144-39.0144-39.0144 39.0144-77.9776-78.1312 78.0288-78.0288-78.0288-77.9776-72.448 72.448-83.5584-100.3008 156.0064-128z" fill="#c83939ff" p-id="24626">
								</path>
								</svg>
								{{ record.info.name }}
                </div>
            </div>
<div v-else-if="column.dataIndex === 'dataset'">
  <div class="text-subtext">
    {{ record.resultData[0] }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'Poisoner'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[1]" style="color: blue;">
    {{ convert(record.resultData[1], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[1], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'before'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[2]" style="color: blue;">
    {{ convert(record.resultData[2], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[2], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'after'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[3]" style="color: blue;">
    {{ convert(record.resultData[3], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[3], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'PPL'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[4]" style="color: blue;">
    {{ convert(record.resultData[4], 2) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[4], 2) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'USE'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[5]" style="color: blue;">
    {{ convert(record.resultData[5], 2) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[5], 2) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'GRAMMAR'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[6]" style="color: blue;">
    {{ convert(record.resultData[6], 2) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[6], 2) }}
  </div>
</div>
            <div v-else class="text-subtext">
                {{ record.resultData[column.dataIndex] }}
            </div>
        </template>
    </a-table>

    <a-table v-if="responseData.RLMI.length > 0" v-bind="$attrs" 
        :columns="RLMIAttackColumns" :dataSource="responseData.RLMI" 
        :pagination="false" class="outer-table privacy-attack-table" :rowClassName="rowClassName">
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
<svg v-if="record.info.defenderEnabled" style="font-size: 12px;" t="1753282514052" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="26441" width="200" height="200"><path d="M260.019513 846.71367C122.197171 732.496698 4.0592 531.652621 4.0592 216.591573A781.279013 781.279013 0 0 1 27.696319 11.810622a1210.477661 1210.477661 0 0 0 252.023439 35.447742A687.889375 687.889375 0 0 0 508.106079 0a725.464298 725.464298 0 0 0 232.339069 47.258364c110.264223 0 252.039314-35.447741 252.039314-35.447742A916.196323 916.196323 0 0 1 1020.010832 212.718197c0 319.077295-102.374601 515.921-240.212817 630.122097-82.706105 66.942734-271.723685 181.159706-271.723685 181.159706s-173.238334-110.327721-248.054817-177.28633z m39.368741-59.08486a2289.371655 2289.371655 0 0 0 208.543206 149.553592V508.174247h395.973335a1053.875794 1053.875794 0 0 0 37.38443-291.5668 816.060025 816.060025 0 0 0-7.937246-114.216972 998.616687 998.616687 0 0 1-196.843705 23.637119A757.340278 757.340278 0 0 1 508.106079 82.706105h-0.174619v425.436393h-390.512511a603.421201 603.421201 0 0 0 181.969305 279.502186z" fill="#1C87FF" p-id="26442">
</path>
</svg>
								<svg v-else style="font-size: 12px;" t="1753281965668" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="24625" width="200" height="200">
								<path d="M771.1744 208.4352l-44.5952 139.3152-150.4256 122.5728-33.4336 27.8528 27.8528 33.4336 83.5584 105.8816 33.4336 39.0144 33.4336-33.4336 44.5952-44.5952 16.6912 16.6912-50.1248 44.5952-33.4336 33.4336 33.4336 33.4336 78.0288 77.9776 5.5808 5.5808-16.6912 16.6912-5.5808-5.5808-78.0288-77.9776-33.4336-33.4336-27.8528 27.8528-44.5952 44.5952-16.6912-16.6912 44.5952-44.5952 33.4336-33.4336-39.0144-27.8528-116.992-94.72-27.8528-22.272-27.8528 22.272-116.992 94.72-39.0144 33.4336 33.4336 33.4336 44.5952 44.5952-16.6912 16.6912-44.5952-50.1248-33.4336-33.4336-33.4336 33.4336-78.0288 78.0288-5.5808 5.5808-16.6912-16.6912 5.5808-5.5808 78.0288-78.0288 33.4336-33.4336-27.8528-27.8528-44.5952-50.1248 16.6912-16.6912 44.5952 44.5952 33.4336 33.4336 27.8528-33.4336 83.5584-100.3008 27.8528-39.0144-33.4336-27.8528-150.4256-122.5728-44.5952-139.3152 139.3152 44.5952 111.36 133.7344 33.4336 44.5952 33.4336-44.5952 111.4112-133.7344 139.3152-44.5952m72.448-72.5504L609.5872 214.016 487.0144 358.8608 369.8688 219.5456 130.4576 135.9872l78.0288 234.0352 156.0064 128-78.0288 111.5648-72.448-78.0288-83.5584 78.0288 78.0288 78.0288-78.0288 78.1312-39.0144-39.0144-39.0144 39.0144 156.0064 156.0064 39.0144-39.0144-39.0144-39.0144 78.0288-77.9776 78.0288 78.0288 77.9776-78.0288-72.6016-72.6016 116.992-94.72 116.992 94.72-72.2432 72.6016 78.0288 78.0288 77.9776-78.0288 78.0288 78.0288-39.0144 39.0144 39.0144 39.0144 156.0064-156.0576-39.0144-39.0144-39.0144 39.0144-77.9776-78.1312 78.0288-78.0288-78.0288-77.9776-72.448 72.448-83.5584-100.3008 156.0064-128z" fill="#c83939ff" p-id="24626">
								</path>
								</svg>
								{{ record.info.name }}
                </div>
            </div>
<div v-else-if="column.dataIndex === 'ASR_Attack'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[0]" style="color: blue;">
    {{ convert(record.resultData[0], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[0], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'WER_Attack'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[1]" style="color: blue;">
    {{ convert(record.resultData[1], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[1], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'ASR_Inference'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[2]" style="color: blue;">
    {{ convert(record.resultData[2], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[2], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'WER_Inference'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[3]" style="color: blue;">
    {{ convert(record.resultData[3], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData[3], 2, true) }}
  </div>
</div>
            <div v-else class="text-subtext">
                {{ record.resultData[column.dataIndex] }}
            </div>
        </template>
    </a-table>

    <a-table v-if="responseData.FET.length > 0" v-bind="$attrs" 
        :columns="FETAttackColumns" :dataSource="responseData.FET" 
        :pagination="false" @expand="getInnerData" class="outer-table privacy-attack-table" :rowClassName="rowClassName">
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
<svg v-if="record.info.defenderEnabled" style="font-size: 12px;" t="1753282514052" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="26441" width="200" height="200"><path d="M260.019513 846.71367C122.197171 732.496698 4.0592 531.652621 4.0592 216.591573A781.279013 781.279013 0 0 1 27.696319 11.810622a1210.477661 1210.477661 0 0 0 252.023439 35.447742A687.889375 687.889375 0 0 0 508.106079 0a725.464298 725.464298 0 0 0 232.339069 47.258364c110.264223 0 252.039314-35.447741 252.039314-35.447742A916.196323 916.196323 0 0 1 1020.010832 212.718197c0 319.077295-102.374601 515.921-240.212817 630.122097-82.706105 66.942734-271.723685 181.159706-271.723685 181.159706s-173.238334-110.327721-248.054817-177.28633z m39.368741-59.08486a2289.371655 2289.371655 0 0 0 208.543206 149.553592V508.174247h395.973335a1053.875794 1053.875794 0 0 0 37.38443-291.5668 816.060025 816.060025 0 0 0-7.937246-114.216972 998.616687 998.616687 0 0 1-196.843705 23.637119A757.340278 757.340278 0 0 1 508.106079 82.706105h-0.174619v425.436393h-390.512511a603.421201 603.421201 0 0 0 181.969305 279.502186z" fill="#1C87FF" p-id="26442">
</path>
</svg>
								<svg v-else style="font-size: 12px;" t="1753281965668" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="24625" width="200" height="200">
								<path d="M771.1744 208.4352l-44.5952 139.3152-150.4256 122.5728-33.4336 27.8528 27.8528 33.4336 83.5584 105.8816 33.4336 39.0144 33.4336-33.4336 44.5952-44.5952 16.6912 16.6912-50.1248 44.5952-33.4336 33.4336 33.4336 33.4336 78.0288 77.9776 5.5808 5.5808-16.6912 16.6912-5.5808-5.5808-78.0288-77.9776-33.4336-33.4336-27.8528 27.8528-44.5952 44.5952-16.6912-16.6912 44.5952-44.5952 33.4336-33.4336-39.0144-27.8528-116.992-94.72-27.8528-22.272-27.8528 22.272-116.992 94.72-39.0144 33.4336 33.4336 33.4336 44.5952 44.5952-16.6912 16.6912-44.5952-50.1248-33.4336-33.4336-33.4336 33.4336-78.0288 78.0288-5.5808 5.5808-16.6912-16.6912 5.5808-5.5808 78.0288-78.0288 33.4336-33.4336-27.8528-27.8528-44.5952-50.1248 16.6912-16.6912 44.5952 44.5952 33.4336 33.4336 27.8528-33.4336 83.5584-100.3008 27.8528-39.0144-33.4336-27.8528-150.4256-122.5728-44.5952-139.3152 139.3152 44.5952 111.36 133.7344 33.4336 44.5952 33.4336-44.5952 111.4112-133.7344 139.3152-44.5952m72.448-72.5504L609.5872 214.016 487.0144 358.8608 369.8688 219.5456 130.4576 135.9872l78.0288 234.0352 156.0064 128-78.0288 111.5648-72.448-78.0288-83.5584 78.0288 78.0288 78.0288-78.0288 78.1312-39.0144-39.0144-39.0144 39.0144 156.0064 156.0064 39.0144-39.0144-39.0144-39.0144 78.0288-77.9776 78.0288 78.0288 77.9776-78.0288-72.6016-72.6016 116.992-94.72 116.992 94.72-72.2432 72.6016 78.0288 78.0288 77.9776-78.0288 78.0288 78.0288-39.0144 39.0144 39.0144 39.0144 156.0064-156.0576-39.0144-39.0144-39.0144 39.0144-77.9776-78.1312 78.0288-78.0288-78.0288-77.9776-72.448 72.448-83.5584-100.3008 156.0064-128z" fill="#c83939ff" p-id="24626">
								</path>
								</svg>
								{{ record.info.name }}
                </div>
            </div>
<div v-else-if="column.dataIndex === 'rouge1'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[0]" style="color: blue;">
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[0], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[0], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'rouge2'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[1]" style="color: blue;">
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[1], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[1], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'rougeL'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[2]" style="color: blue;">
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[2], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[2], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'wrr'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[4]" style="color: blue;">
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[4], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[4], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'distance'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[5]" style="color: blue;">
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[5], 2) }}
  </div>
  <div class="text-subtext" v-else>
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[5], 2) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'fr'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[3]" style="color: blue;">
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[3], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ record.resultData.at(-1) && convert(record.resultData.at(-1)[3], 2, true) }}
  </div>
</div>
            <div v-else class="text-subtext">
                {{ record.resultData[column.dataIndex] }}
            </div>
        </template>
				<template #expandIcon="{ expanded, onExpand, record }">
				    <a @click.stop="e => onExpand(record, e)" class="custom-expand-icon" style="margin-left: 5px;" :title="expanded ? '收起详情' : '展开详情'" >
			<i v-if="!expanded" class="fa-solid fa-caret-down fa-xl"></i>
			<i v-else class="fa-solid fa-caret-up fa-xl"></i>
    </a>
				</template>

        <template #expandedRowRender>
            <a-table v-if="responseData.FET.length > 0"
                :columns="FETInnoColumns" :data-source="FETInnerData" 
                :pagination="false" class="inner-table">
                <template #bodyCell="{ column, record }">
                    <div class="" v-if="column.dataIndex === 'rouge1'">
                        <div class="text-subtext">
                            {{ convert(record[2], 2, true) }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'rouge2'">
                        <div class="text-subtext">
                            {{ convert(record[3], 2, true) }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'rougeL'">
                        <div class="text-subtext">
                            {{ convert(record[4], 2, true) }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'wrr'">
                        <div class="text-subtext">
                            {{ convert(record[5], 2, true) }}
                        </div>
                    </div>
                    <div class="" v-else-if="column.dataIndex === 'distance'">
                        <div class="text-subtext">
                            {{ convert(record[6], 2) }}
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
        :pagination="false" class="outer-table privacy-attack-table" :rowClassName="rowClassName">
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
                <h4>模型窃取</h4>
            </div>
        </template>
        <template #bodyCell="{ column, record }">
            <div class="" v-if="column.dataIndex === 'index'">
                <div class="text-subtext">
<svg v-if="record.info.defenderEnabled" style="font-size: 12px;" t="1753282514052" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="26441" width="200" height="200"><path d="M260.019513 846.71367C122.197171 732.496698 4.0592 531.652621 4.0592 216.591573A781.279013 781.279013 0 0 1 27.696319 11.810622a1210.477661 1210.477661 0 0 0 252.023439 35.447742A687.889375 687.889375 0 0 0 508.106079 0a725.464298 725.464298 0 0 0 232.339069 47.258364c110.264223 0 252.039314-35.447741 252.039314-35.447742A916.196323 916.196323 0 0 1 1020.010832 212.718197c0 319.077295-102.374601 515.921-240.212817 630.122097-82.706105 66.942734-271.723685 181.159706-271.723685 181.159706s-173.238334-110.327721-248.054817-177.28633z m39.368741-59.08486a2289.371655 2289.371655 0 0 0 208.543206 149.553592V508.174247h395.973335a1053.875794 1053.875794 0 0 0 37.38443-291.5668 816.060025 816.060025 0 0 0-7.937246-114.216972 998.616687 998.616687 0 0 1-196.843705 23.637119A757.340278 757.340278 0 0 1 508.106079 82.706105h-0.174619v425.436393h-390.512511a603.421201 603.421201 0 0 0 181.969305 279.502186z" fill="#1C87FF" p-id="26442">
</path>
</svg>
								<svg v-else style="font-size: 12px;" t="1753281965668" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="24625" width="200" height="200">
								<path d="M771.1744 208.4352l-44.5952 139.3152-150.4256 122.5728-33.4336 27.8528 27.8528 33.4336 83.5584 105.8816 33.4336 39.0144 33.4336-33.4336 44.5952-44.5952 16.6912 16.6912-50.1248 44.5952-33.4336 33.4336 33.4336 33.4336 78.0288 77.9776 5.5808 5.5808-16.6912 16.6912-5.5808-5.5808-78.0288-77.9776-33.4336-33.4336-27.8528 27.8528-44.5952 44.5952-16.6912-16.6912 44.5952-44.5952 33.4336-33.4336-39.0144-27.8528-116.992-94.72-27.8528-22.272-27.8528 22.272-116.992 94.72-39.0144 33.4336 33.4336 33.4336 44.5952 44.5952-16.6912 16.6912-44.5952-50.1248-33.4336-33.4336-33.4336 33.4336-78.0288 78.0288-5.5808 5.5808-16.6912-16.6912 5.5808-5.5808 78.0288-78.0288 33.4336-33.4336-27.8528-27.8528-44.5952-50.1248 16.6912-16.6912 44.5952 44.5952 33.4336 33.4336 27.8528-33.4336 83.5584-100.3008 27.8528-39.0144-33.4336-27.8528-150.4256-122.5728-44.5952-139.3152 139.3152 44.5952 111.36 133.7344 33.4336 44.5952 33.4336-44.5952 111.4112-133.7344 139.3152-44.5952m72.448-72.5504L609.5872 214.016 487.0144 358.8608 369.8688 219.5456 130.4576 135.9872l78.0288 234.0352 156.0064 128-78.0288 111.5648-72.448-78.0288-83.5584 78.0288 78.0288 78.0288-78.0288 78.1312-39.0144-39.0144-39.0144 39.0144 156.0064 156.0064 39.0144-39.0144-39.0144-39.0144 78.0288-77.9776 78.0288 78.0288 77.9776-78.0288-72.6016-72.6016 116.992-94.72 116.992 94.72-72.2432 72.6016 78.0288 78.0288 77.9776-78.0288 78.0288 78.0288-39.0144 39.0144 39.0144 39.0144 156.0064-156.0576-39.0144-39.0144-39.0144 39.0144-77.9776-78.1312 78.0288-78.0288-78.0288-77.9776-72.448 72.448-83.5584-100.3008 156.0064-128z" fill="#c83939ff" p-id="24626">
								</path>
								</svg>
								{{ record.info.name }}
                </div>
            </div>
<div v-else-if="column.dataIndex === 'train_loss'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[0]" style="color: blue;">
    {{ convert(record.resultData.iteration.at(-1)[0], 2) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData.iteration.at(-1)[0], 2) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'train_acc'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[1]" style="color: blue;">
    {{ convert(record.resultData.iteration.at(-1)[1], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData.iteration.at(-1)[1], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'valid_acc'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[2]" style="color: blue;">
    {{ convert(record.resultData.iteration.at(-1)[2], 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData.iteration.at(-1)[2], 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'victim_acc'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[3]" style="color: blue;">
    {{ convert(record.resultData.victim_acc, 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData.victim_acc, 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'steal_acc'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[4]" style="color: blue;">
    {{ convert(record.resultData.steal_acc, 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData.steal_acc, 2, true) }}
  </div>
</div>
<div v-else-if="column.dataIndex === 'agreement'">
  <div class="text-subtext" v-if="record.compResult && record.compResult[5]" style="color: blue;">
    {{ convert(record.resultData.agreement, 2, true) }}
  </div>
  <div class="text-subtext" v-else>
    {{ convert(record.resultData.agreement, 2, true) }}
  </div>
</div>
        </template>
    </a-table>
</div>
    `,

	setup(props) { // `props` is automatically passed here
		// Constants for columns
		const NormalTrainColumns = [
			{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
			{ title: '分类准确率\n(Classification Accuracy,%)', dataIndex: 'acc', align: 'center' },
			{ title: 'F1分数\n(F1 Score)', dataIndex: 'f1', align: 'center' },
		];
		const AdvColumns = [
			{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
			{ title: '攻击成功次数\n(Successful Attacks,次)', dataIndex: 'success', align: 'center' },
			{ title: '攻击失败次数\n(Failed Attacks,次)', dataIndex: 'fail', align: 'center' },
			{ title: '跳过攻击次数\n(Skipped Attacks,次)', dataIndex: 'skip', align: 'center' },
			{ title: '原始样本准确率\n(Clean Accuracy,%)', dataIndex: 'before', align: 'center' },
			{ title: '对抗样本准确率\n(Adversarial Accuracy,%)', dataIndex: 'after', align: 'center' },
			{ title: '攻击成功率\n(Attack Success Rate,%)', dataIndex: 'rate', align: 'center' },
		];
		const PoisoningColumns = [
			{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
			{ title: '毒化数据集准确率\n(Poisoned Accuracy,%)', dataIndex: 'acc', align: 'center' },
			{ title: 'F1分数\n(F1 Score)', dataIndex: 'f1', align: 'center' },
		];
		const BackDoorColumns = [
			{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
			{ title: '后门数据集\n(Poisoned Dataset,名称)', dataIndex: 'dataset', align: 'center' },
			// { title: '投毒方', dataIndex: 'Poisoner', align: 'center' },
			{ title: '原始数据集准确率\n(Clean Accuracy,%)', dataIndex: 'before', align: 'center' },
			{ title: '后门数据集准确率\n(Poisoned Accuracy,%)', dataIndex: 'after', align: 'center' },
			{ title: '困惑度\n(PPL-Perplexity)', dataIndex: 'PPL', align: 'center' },
			{ title: '语义相似性\n(USE-Universal Sentence Encoder Similarity)', dataIndex: 'USE', align: 'center' },
			{ title: '语法正确性得分\n(Grammar Score)', dataIndex: 'GRAMMAR', align: 'center' },
		];
		const RLMIAttackColumns = [
			{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
			{ title: '攻击阶段成功率\n(Attack Success Rate(Attack Phase), %)', dataIndex: 'ASR_Attack', align: 'center' },
			{ title: '攻击阶段词错误率\n(Word Error Rate (Attack Phase), %)', dataIndex: 'WER_Attack', align: 'center' },
			{ title: '推理阶段成功率\n(Attack Success Rate (Inference Phase), %)', dataIndex: 'ASR_Inference', align: 'center' },
			{ title: '推理阶段词错误率\n(Word Error Rate (Inference Phase), %)', dataIndex: 'WER_Inference', align: 'center' },
		];
		const FETAttackColumns = [
			{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
			{ title: 'ROUGE-1得分\n(ROUGE-1 Score,%)', dataIndex: 'rouge1', align: 'center' },
			{ title: 'ROUGE-2得分\n(ROUGE-2 Score,%)', dataIndex: 'rouge2', align: 'center' },
			{ title: 'ROUGE-L得分\n(ROUGE-L Score,%)', dataIndex: 'rougeL', align: 'center' },
			{ title: '词汇恢复率\n(Token Recovery Rate,%)', dataIndex: 'wrr', align: 'center' },
			{ title: '编辑距离\n(Edit Distance)', dataIndex: 'distance', align: 'center' },
			{ title: '完全恢复率\n(Exact Match Rate,%)', dataIndex: 'fr', align: 'center' },
		];
		const FETInnoColumns = [
			{ title: 'ROUGE-1得分\n(ROUGE-1 Score,%)', dataIndex: 'rouge1', align: 'center' },
			{ title: 'ROUGE-2得分\n(ROUGE-2 Score,%)', dataIndex: 'rouge2', align: 'center' },
			{ title: 'ROUGE-L得分\n(ROUGE-L Score,%)', dataIndex: 'rougeL', align: 'center' },
			{ title: '词汇恢复率\n(Token Recovery Rate,%)', dataIndex: 'wrr', align: 'center' },
			{ title: '编辑距离\n(Edit Distance)', dataIndex: 'distance', align: 'center' },
			{ title: '完全恢复', dataIndex: 'fr', align: 'center' },
		];
		const ModelStealingAttackColumns = [
			{ title: '实验配置', dataIndex: 'index', width: '10%', align: 'center' },
			{ title: '训练集准确率\n(Training Accuracy,%)', dataIndex: 'train_acc', align: 'center' },
			{ title: '训练损失\n(Training Loss)', dataIndex: 'train_loss', align: 'center' },
			{ title: '验证集准确率\n(Validation Accuracy,%)', dataIndex: 'valid_acc', align: 'center' },
			{ title: '目标模型准确率\n(Target Model Accuracy,%)', dataIndex: 'victim_acc', align: 'center' },
			{ title: '替代模型准确率\n(Stolen Model Accuracy,%)', dataIndex: 'steal_acc', align: 'center' },
			{ title: '一致性分数\n(Agreement,%)', dataIndex: 'agreement', align: 'center' },
		];

		const responseData = ref({
			NormalTrain: [],
			AdvAttack: [],
			BackdoorAttack: [],
			PoisoningAttack: [],
			RLMI: [],
			FET: [],
			ModelStealingAttack: []
		});

		// 子表格
		const FETInnerData = ref([]);

		const username = localStorage.getItem('Global_username');
		const token = localStorage.getItem('Global_token');

		function getInnerData(expanded, record) {
			if (expanded) {
				FETInnerData.value = record.resultData.slice(0, -1); // 排除最后一个
				console.log("Expanded FET inner data:", FETInnerData.value);
			}
		}

		function convert(str, decimalPlaces, percent) {
			// if (decimalPlaces !== undefined) 
			// 	return str;

			if (str === 'nan')
				return '未测试';

			// 检查是否为百分数
			if (str.includes('%')) {
				const valueWithoutPercent = parseFloat(str) / 100;
				// 如果提供了 decimalPlaces，保留指定位数
				if (decimalPlaces !== undefined) {
            return `${(valueWithoutPercent * 100).toFixed(decimalPlaces)}%`;
				}
				return valueWithoutPercent;
			}

			// 对于小数和整数
			const num = parseFloat(str);

			// 检查是否是有效的数字
			if (isNaN(num)) {
				console.warn(`"${str}" 无法转换为有效数字。`);
				return null;
			}

    // 如果提供了 decimalPlaces
    if (decimalPlaces !== undefined) {
        // 如果 percent 为 true，返回百分数字符串
        if (percent === true) {
            return `${(num * 100).toFixed(decimalPlaces)}%`;
        }
        // 如果不是整数，保留指定位数
        if (!Number.isInteger(num)) {
            return Number(num.toFixed(decimalPlaces));
        }
    }

			// 默认返回原始数字
			return num;
		}
		// AdvAttack: [],
		// BackdoorAttack: [],
		// PoisoningAttack: [],
		// RLMI: [],
		// FET: [],
		// ModelStealingAttack: []

		function compare(e1, e2) {
			let arr;
			switch (e1.info.type) {
				case 'AdvAttack':
					arr = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]];
					if (convert(e1.resultData[0]) > convert(e2.resultData[0]))
						arr[1][0] = 1;
					else if (convert(e1.resultData[0]) < convert(e2.resultData[0]))
						arr[0][0] = 1;

					if (convert(e1.resultData[1]) > convert(e2.resultData[1]))
						arr[0][1] = 1;
					else if (convert(e1.resultData[1]) < convert(e2.resultData[1]))
						arr[1][1] = 1;

					if (convert(e1.resultData[2]) > convert(e2.resultData[2]))
						arr[0][2] = 1;
					else if (convert(e1.resultData[2]) < convert(e2.resultData[2]))
						arr[1][2] = 1;

					if (convert(e1.resultData[3]) > convert(e2.resultData[3]))
						arr[0][3] = 1;
					else if (convert(e1.resultData[3]) < convert(e2.resultData[3]))
						arr[1][3] = 1;

					if (convert(e1.resultData[4]) > convert(e2.resultData[4]))
						arr[0][4] = 1;
					else if (convert(e1.resultData[4]) < convert(e2.resultData[4]))
						arr[1][4] = 1;

					if (convert(e1.resultData[5]) > convert(e2.resultData[5]))
						arr[1][5] = 1;
					else if (convert(e1.resultData[5]) < convert(e2.resultData[5]))
						arr[0][5] = 1;

					e1.compResult = arr[0];
					e2.compResult = arr[1];
					break;

				case 'BackdoorAttack':
					arr = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]];
					if (convert(e1.resultData[2]) > convert(e2.resultData[2]))
						arr[0][2] = 1;
					else if (convert(e1.resultData[2]) < convert(e2.resultData[2]))
						arr[1][2] = 1;

					if (convert(e1.resultData[3]) > convert(e2.resultData[3]))
						arr[1][3] = 1;
					else if (convert(e1.resultData[3]) < convert(e2.resultData[3]))
						arr[0][3] = 1;

					if (convert(e1.resultData[4]) > convert(e2.resultData[4]))
						arr[0][4] = 1;
					else if (convert(e1.resultData[4]) < convert(e2.resultData[4]))
						arr[1][4] = 1;

					if (convert(e1.resultData[5]) > convert(e2.resultData[5]))
						arr[1][5] = 1;
					else if (convert(e1.resultData[5]) < convert(e2.resultData[5]))
						arr[0][5] = 1;

					if (convert(e1.resultData[6]) > convert(e2.resultData[6]))
						arr[1][6] = 1;
					else if (convert(e1.resultData[6]) < convert(e2.resultData[6]))
						arr[0][6] = 1;

					e1.compResult = arr[0];
					e2.compResult = arr[1];
					break;

				case "PoisoningAttack":
					arr = [[0, 0], [0, 0]];
					if (convert(e1.resultData[0]) > convert(e2.resultData[0]))
						arr[0][0] = 1;
					else if (convert(e1.resultData[0]) < convert(e2.resultData[0]))
						arr[1][0] = 1;

					if (convert(e1.resultData[1]) > convert(e2.resultData[1]))
						arr[0][1] = 1;
					else if (convert(e1.resultData[1]) < convert(e2.resultData[1]))
						arr[1][1] = 1;
					e1.compResult = arr[0];
					e2.compResult = arr[1];
					break;

				case "RLMI":
					arr = [[0, 0, 0, 0], [0, 0, 0, 0]];
					if (convert(e1.resultData[0]) > convert(e2.resultData[0]))
						arr[1][0] = 1;
					else if (convert(e1.resultData[0]) < convert(e2.resultData[0]))
						arr[0][0] = 1;

					if (convert(e1.resultData[1]) > convert(e2.resultData[1]))
						arr[0][1] = 1;
					else if (convert(e1.resultData[1]) < convert(e2.resultData[1]))
						arr[1][1] = 1;

					if (convert(e1.resultData[2]) > convert(e2.resultData[2]))
						arr[1][2] = 1;
					else if (convert(e1.resultData[2]) < convert(e2.resultData[2]))
						arr[0][2] = 1;

					if (convert(e1.resultData[3]) > convert(e2.resultData[3]))
						arr[0][3] = 1;
					else if (convert(e1.resultData[3]) < convert(e2.resultData[3]))
						arr[1][3] = 1;

					e1.compResult = arr[0];
					e2.compResult = arr[1];
					break;

				case "FET":
					arr = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]];
					if (convert(e1.resultData.at(-1)[0]) > convert(e2.resultData.at(-1)[0]))
						arr[1][0] = 1;
					else if (convert(e1.resultData.at(-1)[0]) < convert(e2.resultData.at(-1)[0]))
						arr[0][0] = 1;

					if (convert(e1.resultData.at(-1)[1]) > convert(e2.resultData.at(-1)[1]))
						arr[1][1] = 1;
					else if (convert(e1.resultData.at(-1)[1]) < convert(e2.resultData.at(-1)[1]))
						arr[0][1] = 1;

					if (convert(e1.resultData.at(-1)[2]) > convert(e2.resultData.at(-1)[2]))
						arr[1][2] = 1;
					else if (convert(e1.resultData.at(-1)[2]) < convert(e2.resultData.at(-1)[2]))
						arr[0][2] = 1;

					if (convert(e1.resultData.at(-1)[3]) > convert(e2.resultData.at(-1)[3]))
						arr[1][3] = 1;
					else if (convert(e1.resultData.at(-1)[3]) < convert(e2.resultData.at(-1)[3]))
						arr[0][3] = 1;

					if (convert(e1.resultData.at(-1)[4]) > convert(e2.resultData.at(-1)[4]))
						arr[0][4] = 1;
					else if (convert(e1.resultData.at(-1)[4]) < convert(e2.resultData.at(-1)[4]))
						arr[1][4] = 1;

					if (convert(e1.resultData.at(-1)[5]) > convert(e2.resultData.at(-1)[5]))
						arr[1][5] = 1;
					else if (convert(e1.resultData.at(-1)[5]) < convert(e2.resultData.at(-1)[5]))
						arr[0][5] = 1;

					e1.compResult = arr[0];
					e2.compResult = arr[1];
					break;

				case "ModelStealingAttack":
					arr = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]];
					if (convert(e1.resultData.iteration.at(-1)[0]) > convert(e2.resultData.iteration.at(-1)[0]))
						arr[0][0] = 1;
					else if (convert(e1.resultData.iteration.at(-1)[0]) < convert(e2.resultData.iteration.at(-1)[0]))
						arr[1][0] = 1;

					if (convert(e1.resultData.iteration.at(-1)[1]) > convert(e2.resultData.iteration.at(-1)[1]))
						arr[1][1] = 1;
					else if (convert(e1.resultData.iteration.at(-1)[1]) < convert(e2.resultData.iteration.at(-1)[1]))
						arr[0][1] = 1;

					if (convert(e1.resultData.iteration.at(-1)[2]) > convert(e2.resultData.iteration.at(-1)[2]))
						arr[0][2] = 1;
					else if (convert(e1.resultData.iteration.at(-1)[2]) < convert(e2.resultData.iteration.at(-1)[2]))
						arr[1][2] = 1;

					if (convert(e1.resultData.victim_acc) > convert(e2.resultData.victim_acc))
						arr[0][3] = 1;
					else if (convert(e1.resultData.victim_acc) < convert(e2.resultData.victim_acc))
						arr[1][3] = 1;

					if (convert(e1.resultData.steal_acc) > convert(e2.resultData.steal_acc))
						arr[1][4] = 1;
					else if (convert(e1.resultData.steal_acc) < convert(e2.resultData.steal_acc))
						arr[0][4] = 1;

					if (convert(e1.resultData.agreement) > convert(e2.resultData.agreement))
						arr[1][5] = 1;
					else if (convert(e1.resultData.agreement) < convert(e2.resultData.agreement))
						arr[0][5] = 1;

					e1.compResult = arr[0];
					e2.compResult = arr[1];
					break;

				default:
					break;
			}

			return [e1, e2];
		}
		// 处理函数
		function processControlGroup(attackList) {
			if (attackList.length === 0)
				return attackList;

			let res = [];
			let last = [];
			// 使用 Map 存储 buffer，键是 info.name，值是对应的元素
			let bufferMap = new Map();

			const len = attackList.length;
			for (let i = 0; i < len; ++i) {
				let e = attackList[i];
				if ((e.info !== undefined) && e.info.comparison) {
					const comparisonName = e.info.comparison.name;
					if (bufferMap.has(comparisonName)) {
						let matchedElement = bufferMap.get(comparisonName);
						if (e.info.defenderEnabled === false) {
							res.push(...(compare(e, matchedElement)));
						} else {
							res.push(...(compare(matchedElement, e)));
						}
						bufferMap.delete(comparisonName); // 从 Map 中移除已配对的
					} else {
						// 如果当前元素还没有找到它的对照，就把它放入 bufferMap
						// 以它自己的 name 作为键，等待其对照组出现
						bufferMap.set(e.info.name, e);
					}
				} else {
					last.push(e);
				}
			}

			if (bufferMap.keys.length > 0) {
				console.warn("存在含有comparison的配置，但是找不到对照组。")
				for (let item of bufferMap.values())
					last.push(item);
			}

			res.push(...last);

			return res;
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
				console.log("API response:", response);
				if (response && response.data && typeof response.data === 'object') {
					// Assign specific keys if they exist, or default to empty arrays
					responseData.value.NormalTrain = response.data.result.normalTrain || [];
					responseData.value.AdvAttack = processControlGroup(response.data.result.AdvAttack || []);
					responseData.value.BackdoorAttack = processControlGroup(response.data.result.BackdoorAttack || []);
					responseData.value.PoisoningAttack = processControlGroup(response.data.result.PoisoningAttack || []);
					responseData.value.RLMI = processControlGroup(response.data.result.RLMI || []);
					responseData.value.FET = processControlGroup(response.data.result.FET || []);
					responseData.value.ModelStealingAttack = processControlGroup(response.data.result.ModelStealingAttack || []);
				} else {
					console.error("Invalid API response structure:", response);
				}
				console.log("Fetched responseData:", responseData.value);
			} catch (error) {
				console.error("请求出错 (Error fetching data):", error);
				// Reset data or show error message to user
				responseData.value = {
					NormalTrain: [],
					AdvAttack: [],
					BackdoorAttack: [],
					PoisoningAttack: [],
					RLMI: [],
					FET: [],
					ModelStealingAttack: []
				};
			}

			// 预处理各个结果,添加索引什么的
			// if (responseData.value.AdvAttack && Array.isArray(responseData.value.AdvAttack)) {
			//     responseData.value.AdvAttack = responseData.value.AdvAttack.map((item, index) => {
			//         return [[index + 1, ...item]];
			//     });
			// }
			// if (responseData.value.PoisoningAttack && Array.isArray(responseData.value.PoisoningAttack)) {
			//     responseData.value.PoisoningAttack = responseData.value.PoisoningAttack.map((item, index) => {
			//         return [index + 1, ...item]; // This creates [index, item_val1, item_val2, ...]
			//     });
			// }
			// if (responseData.value.BackdoorAttack && Array.isArray(responseData.value.BackdoorAttack)) {
			//     responseData.value.BackdoorAttack = responseData.value.BackdoorAttack.map((item, index) => {
			//         return [index + 1, ...item];
			//     });
			// }
			// if (responseData.value.RLMI && Array.isArray(responseData.value.RLMI)) {
			//     responseData.value.RLMI = responseData.value.RLMI.map((item, index) => {
			//         return [index + 1, ...item];
			//     });
			// }
			// if (responseData.value.FET && Array.isArray(responseData.value.FET)) {
			//     responseData.value.FET = responseData.value.FET.map((item, index) => {
			//         return item.map((element) => {
			//             if (element.length == 6) {
			//                 return [index + 1, ...element]
			//             } else {
			//                 return element
			//             }
			//         })
			//     })
			// }
			// if (responseData.value.ModelStealingAttack && Array.isArray(responseData.value.ModelStealingAttack)) {
			//     responseData.value.ModelStealingAttack = responseData.value.ModelStealingAttack.map((item, index) => {
			//         item.index = index + 1; 
			//         return item; 
			//     }); 
			// }
			// console.log("Processed responseData:", responseData.value);
		});

		function rowClassName(record, index) {
			console.log("row: ", record);

			if (record.info.defenderEnabled) {
				return "defender-row";
			} else if (record.info.comparison) {
				return "attack-row";
			} else {
				return "only-attack-row";
			}
		}

		// Return everything that the template needs
		return {
			NormalTrainColumns,
			AdvColumns,
			PoisoningColumns,
			BackDoorColumns,
			RLMIAttackColumns,
			FETAttackColumns,
			FETInnoColumns,
			ModelStealingAttackColumns,
			responseData,
			FETInnerData,
			getInnerData,
			rowClassName,
			convert,
			// username, token, props are available via `this` or closure, but not directly needed by template
		};
	}
});

export default AttackTable;