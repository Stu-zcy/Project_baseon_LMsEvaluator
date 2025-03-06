<script lang="ts" setup>
import axios from 'axios';
import { ref, onMounted } from 'vue';
import { message } from 'ant-design-vue';

const username = localStorage.getItem('Global_username');  // 从 localStorage 获取用户名
const token = localStorage.getItem('Global_token');
const responseData = ref({})
const newUsername = ref<string>(), newGender = ref<number>(), newAge = ref<number>()
const open = ref<boolean>(false)
const confirmLoading = ref<boolean>(false)

// 头像上传
const fileList = ref<object[]>()
const imageUrl = ref<string>()
const loading = ref<boolean>()
const beforeUpload = (file) => {
	loading.value = true
	const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
	const isLt2M = file.size / 1024 / 1024 < 20;
	if (!isJpgOrPng) {
		message.error('You can only upload JPG file!');
	} else if (!isLt2M) {
		message.error('Image must smaller than 20MB!');
	} else {
		// 预览
		const reader = new FileReader();
		reader.onload = (e) => {
			imageUrl.value = e.target.result as string;
		};
		reader.readAsDataURL(file);
	}
	// 阻止默认的上传行为
	loading.value = false
	return false
}
const getBase64Url = (img: Blob, callback) => {
	const reader = new FileReader()
	reader.addEventListener('load', () => callback(reader.result as string))
	reader.readAsDataURL(img)
};
// 使用 SHA-256 计算文件的摘要
const getSHA256 = async (file: Blob) => {
  const salt = crypto.getRandomValues(new Uint8Array(16));  // 随机生成16字节的盐值
  const timestamp = Date.now().toString();
  const arrayBuffer = await file.arrayBuffer();
  const combinedBuffer = new Uint8Array(salt.length + arrayBuffer.byteLength + timestamp.length);
  combinedBuffer.set(salt, 0);
  combinedBuffer.set(new Uint8Array(arrayBuffer), salt.length);
  combinedBuffer.set(new TextEncoder().encode(timestamp), salt.length + arrayBuffer.byteLength);
  // 计算 SHA-256 并转换为十六进制字符串
  const hashBuffer = await crypto.subtle.digest('SHA-256', combinedBuffer);
  return [...new Uint8Array(hashBuffer)].map(b => b.toString(16).padStart(2, '0')).join('');
};
const handleChange = (info) => {
	if (info.file.status == 'uploading') {
		loading.value = true
		return
	} else if (info.file.status == 'done') {
		getBase64Url(info.file.originFileObj, (base64Url) => {
			loading.value = false
			imageUrl.value = base64Url
		})
	} else if (info.file.status == 'error') {
		loading.value = false
		message.error('uploading error')
	}
}
const uploadAvatar = async (file: Blob, type: string) => {
	const currentTime = new Date().toISOString().replace(/[:.]/g, '-');  // 获取当前时间并格式化
	const fileName =  currentTime + (type == 'image/jpeg' ? '.jpg' : '.png');

	//const fileName = (await getSHA256(file)) + (type == 'image/jpeg' ? '.jpg' : '.png')
	const rawUrl = 'https://api.github.com/repos/Stu-zcy/Public_image/contents/'
	const fastUrl = 'https://raw.githubusercontent.com/Stu-zcy/Public_image/refs/heads/main/'
	const token = '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'
	const reader = new FileReader();
	reader.onloadend = async function () {
		const base64Img = reader.result.toString().split(',')[1]
		const response = await fetch(rawUrl+fileName, {
			method: 'PUT',
			headers: {
				'Authorization': 'token ' + token,
				'Accept': 'application/vnd.github.v3+json'
			},
			body: JSON.stringify({
				message: 'Upload avatar',
				content: base64Img,
				branch: 'main'
			})
		})
		const data = await response.json()
		if (data.status == 201) {
			console.log('Image uploaded successfully')
		} else {
			console.error('Upload failed')
		}
	}
	reader.readAsDataURL(file)
	return fastUrl + fileName
}


const showModal = () => {
	console.log("click edit.")
	open.value = true
	confirmLoading.value = false
	newUsername.value = responseData.value['account']['username']
	imageUrl.value = responseData.value['account']['avatar']
	newGender.value = responseData.value['account']['gender']
	newAge.value = responseData.value['account']['age']
}

async function handleOk() {
	console.log("ok")
	confirmLoading.value = true
	const newAvatar = await uploadAvatar(fileList.value[0].originFileObj as Blob, 
													        fileList.value[0].type)
	const response = await axios.post('http://localhost:5000/api/updateInfo', {
		username: username,
		token: token,
		newUsername: newUsername.value,
		newAvatar: newAvatar,
		newGender: newGender.value,
		newAge: newAge.value
	});
	fetchInfo()
	open.value = false
	confirmLoading.value = false
}

async function fetchInfo() {
	try {
		const response = await axios.post('http://localhost:5000/api/profile', {
			username: username,
			token: token,
		});
		responseData.value = response.data.data;
	} catch (error) {
		console.error("请求出错", error);
	}
}

onMounted(async () => {
	fetchInfo()
})

</script>
<template>
	<div class="account flex">
		<a-card hoverable style="width: 20rem; ">
			<div class="avatarContainer rounded-full w-4/5 overflow-hidden mx-auto">
				<img :src="responseData['account']['avatar']" class="h-full w-full" />
			</div>
			<a-divider />
			<ul class="infoList list-none pl-10">
				<li class="infoItem">
					<span class="label w-5">
						用<b style="width: 7px;display: inline-block;" />户<b style="width: 7px;display: inline-block;" />名：
					</span>
					<span>{{ responseData['account']['username'] }}</span>
				</li>
				<li class="infoItem">
					<span class="label w-5">权限等级：</span>
					<span>{{ responseData['role'] }}</span>
				</li>
				<li class="infoItem">
					<span class="label w-5">
						年<b style="width: 28px;display: inline-block;" />龄：
					</span>
					<span>{{ responseData['account']['age'] }}</span>
				</li>
				<li class="infoItem">
					<span class="label w-2">
						性<b style="width: 28px;display: inline-block;" />别：
					</span>
					<span>{{ responseData['account']['gender'] == 0 ? '女' : '男' }}</span>
				</li>
			</ul>
			<template #actions>
				<edit-outlined key="edit" @click="showModal" />
			</template>
			<template>
				<a-modal v-if="open" :visible="true" title="修改信息" :maskClosable="false" @ok="handleOk"
					:confirm-loading="confirmLoading" :closable="false" @cancel="() => { open = false }"
					:centered="true", width="350px">
					<a-form :label-col="{style: {width: '100px'}}" class="w-full mx-auto">
						<a-form-item label="头像">
							<a-upload v-model:file-list="fileList" name="avatar" list-type="picture-card" :show-upload-list="false"
								maxCount="1" :before-upload="beforeUpload" >
								<img :src="imageUrl" class="w-full h-full" />
							</a-upload>
						</a-form-item>
						<a-form-item label="用户名">
							<a-input v-model:value="newUsername" placeholder="输入新用户名" style="width: 100px;"></a-input>
						</a-form-item>
						<a-form-item label="性别">
							<a-radio-group v-model:value="newGender">
								<a-radio-button :value="1">男</a-radio-button>
								<a-radio-button :value="0">女</a-radio-button>
							</a-radio-group>
						</a-form-item>
						<a-form-item label="年龄" :rules="[{ type: 'number', min: 0, max: 99 }]">
							<a-input-number v-model:value="newAge"></a-input-number>
						</a-form-item>
					</a-form>
				</a-modal>
			</template>
		</a-card>
		<div class="stars"></div>
	</div>
</template>

<!-- <style lang="css" scoped>
>>> .uploader {

}

</style> -->