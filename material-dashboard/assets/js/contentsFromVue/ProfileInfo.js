// --- 从全局 Vue 获取 ---
const { ref, onMounted, defineComponent } = Vue;
import axios from "../utils/axiosConfig.js";

const ProfileInfo = defineComponent({
	name: 'ProfileInfo',
	template: `
    <div class="account-profile-editor flex">
      <a-card>
        <div class="avatarContainer rounded-full">
          <img v-if="responseData.account && responseData.account.avatar" :src="responseData.account.avatar" class="h-full w-full" alt="User Avatar" id="avatar"/>
          <div v-else class="avatar-placeholder h-full w-full flex items-center justify-center bg-gray-200 text-gray-500">No Avatar</div>
        </div>
        <a-divider />
        <ul v-if="responseData.account" class="infoList list-none pl-10">
          <li class="infoItem">
            <span class="label w-5">用<span class="w-05em"></span>户<span class="w-05em"></span>名：</span>
            <span>{{ responseData.account.username }}</span>
          </li>
          <li class="infoItem">
            <span class="label w-5">权限等级：</span>
            <span>{{ responseData.role }}</span>
          </li>
          <li class="infoItem">
            <span class="label w-5">年<span class="w-2em"></span>龄：</span>
            <span>{{ responseData.account.age }}</span>
          </li>
          <li class="infoItem">
            <span class="label w-2">性<span class="w-2em"></span>别：</span>
            <span>{{ responseData.account.gender === 0 ? '女' : '男' }}</span>
          </li>
        </ul>
        <div v-else class="info-loading">加载中...</div>
        <template slot="actions" class="ant-card-actions">
          <a-button @click="showModal" class="no-border" style="margin: 15px auto 0px;">
            <i class="fa-solid fa-edit" style="font-size: 24px;"></i>
          </a-button>
        </template>
      </a-card>

      <a-modal
        v-if="open"
        :open="open"
        title="修改信息"
        @ok="handleOk"
        :confirm-loading="confirmLoading"
        :closable="!confirmLoading"
        @cancel="handleModalCancel"
        :centered="true"
        width="auto"
        style="min-width: 350px; max-width: 500px;"
      >
        <a-form :label-col="{ style: { width: '100px' } }" class="w-full mx-auto profile-form">
          <a-form-item>
            <template #label>
              头<span class="w-1em"></span>像
            </template>
            <!-- 将要上传，准备裁剪 -->
            <div v-if="isUploading===true" class=cropper-container>
              <vue-cropper
                ref="cropper"
                :img="imageUrl"
                :original="true"
                :auto-crop="true"
                :auto-crop-width="150"
                :auto-crop-height="150"
                :fixedBox="true"
                :center-box="true"
                :info="true"
                :outputType="png"
              />
            </div>
            <!-- 没有上传意图，显示图片或空白 -->
            <div v-else class="display-avatar-container">
              <a-upload
                v-model:file-list="fileList"
                name="avatar"
                list-type="picture-card"
                :show-upload-list="false"
                :maxCount="1"
                :before-upload="beforeUpload"
                accept="image/jpeg,image/png"
              >
                <div v-if="imageUrl">
                  <img :src="imageUrl" alt="avatar preview" id="avatar-preview" />
                  <svg t="1750181260393" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5979" width="200" height="200">
                    <path d="M810.667 938.667H213.333c-70.57 0-128-57.43-128-128V213.333c0-70.57 57.43-128 128-128h256a42.667 42.667 0 0 1 0 85.334h-256c-23.509 0-42.666 19.157-42.666 42.666v597.334c0 23.552 19.157 42.666 42.666 42.666h597.334a42.667 42.667 0 0 0 42.666-42.666v-256a42.667 42.667 0 1 1 85.334 0v256c0 70.57-57.43 128-128 128z" p-id="5980"></path>
                    <path d="M341.333 725.333a42.71 42.71 0 0 1-40.49-56.149l64-192a42.723 42.723 0 0 1 10.325-16.683l341.333-341.333c50.262-50.347 138.07-50.347 188.331 0a133.333 133.333 0 0 1 0 188.33L563.499 648.833a42.837 42.837 0 0 1-16.683 10.283l-192 64a42.112 42.112 0 0 1-13.483 2.218z m101.291-211.626L408.789 615.21l101.504-33.835 334.208-334.208a47.915 47.915 0 0 0 0-67.67 48.981 48.981 0 0 0-67.669 0L442.624 513.708z" p-id="5981"></path>
                  </svg>
                </div>
                <div v-else>
                  <loading-outlined v-if="loading"></loading-outlined>
                  <plus-outlined v-else></plus-outlined>
                  <div class="ant-upload-text">上传</div>
                </div>
              </a-upload>
            </div>
          </a-form-item>

          <a-form-item>
            <!-- 可以使用插槽代替a-form-item中的label -->
            <template #label>
              用户名
            </template>
            <a-input v-model:value="newUsername" placeholder="输入新用户名" disabled="true" style="width: 180px;"></a-input>
          </a-form-item>
          <a-form-item>
            <template #label>
              性<span class="w-1em"></span>别
            </template>
            <a-radio-group v-model:value="newGender">
              <a-radio :value="1">男</a-radio>
              <a-radio :value="0">女</a-radio>
            </a-radio-group>
          </a-form-item>
          <a-form-item>
            <template #label>
              年<span class="w-1em"></span>龄
            </template>
            <a-input-number v-model:value="newAge" :min="0" :max="120"></a-input-number>
          </a-form-item>
        </a-form>
      </a-modal>
    </div>
  `,
	setup() {
		const username = localStorage.getItem('Global_username');
		const token = localStorage.getItem('Global_token');
		const responseData = ref({}); // Initialize as object
		const newUsername = ref('');
		const newGender = ref(undefined); // Use undefined for unselected radio
		const newAge = ref(undefined);
		const open = ref(false);
		const confirmLoading = ref(false);
		const imageUrl = ref('');
		const loading = ref(false); // For a-upload loading state
		const fileList = ref([]);
		const isUploading = ref(false);
		const cropper = ref(null);

		async function fetchInfo() {
			if (!username || !token) {
				console.error("Username or token is missing from localStorage.");
				// antd.message.error("用户未登录或登录已失效");
				return;
			}
			try {
				const response = await axios.post('http://127.0.0.1:5000/api/profile', {
					username: username,
					token: token,
				});
				console.log(response);
				if (response.status === 200) {
					responseData.value = response.data;
				} else {
					responseData.value = {}; // Ensure it's an object
					console.error("Invalid profile data structure:", response.data);
					antd.message.error("获取用户信息失败，请检查网络连接");
				}
			} catch (error) {
				console.error('获取用户信息请求出错:', error);
				responseData.value = {};
				antd.message.error("获取用户信息失败，请检查网络连接");
			}
		}

		onMounted(fetchInfo);

		//转化为base64编码
		function fileToBase64(file) {
			return new Promise((resolve, reject) => {
				const reader = new FileReader();
				reader.onload = e => {
					resolve(e.target.result);
				};
				reader.onerror = reject;
				reader.readAsDataURL(file);
			});
		}

		const beforeUpload = async (file) => {
			loading.value = true; // a-upload loading
			const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
			const isLt2M = file.size / 1024 / 1024 < 20;

			if (!isJpgOrPng) {
				antd.message.error('只能上传 JPG 或 PNG 格式的图片!');
			} else if (!isLt2M) {
				antd.message.error('图片大小必须小于 20MB!');
			} else {
				imageUrl.value = await fileToBase64(file);
				isUploading.value = true;
			}
			loading.value = false;
			return false;
		};

		const uploadToGithub = async (base64File) => {
			loading.value = true; // Indicate general loading
			const fileName = Date.now().toString() + '.png'; // Use timestamp for unique file name
			const rawUrl = 'https://api.github.com/repos/Stu-zcy/Public_image/contents/';
			const fastUrl = 'https://raw.githubusercontent.com/Stu-zcy/Public_image/main/'; // Corrected path
			const githubToken = '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'; // Sensitive data, should not be hardcoded

			const base64ImgContent = base64File.split(',')[1];

			try {
				const response = await fetch(rawUrl + fileName, {
					method: 'PUT',
					headers: {
						'Authorization': 'token ' + githubToken,
						'Accept': 'application/vnd.github.v3+json',
						'Content-Type': 'application/json',
					},
					body: JSON.stringify({
						message: 'Upload cropped avatar for ',
						content: base64ImgContent,
						branch: 'main',
					}),
				});

				loading.value = false;
				if (response.ok) { // Check for 200-299 status
					const responseData = await response.json();
					console.log('裁剪后的图片上传GitHub成功', responseData);
					return fastUrl + fileName;
				} else {
					const errorData = await response.text();
					console.error('GitHub上传失败:', response.status, errorData);
					antd.message.error(`GitHub上传失败: ${response.status}`);
					return '';
				}
			} catch (error) {
				console.error('GitHub上传异常:', error);
				antd.message.error('GitHub上传异常');
				loading.value = false;
				return '';
			}
		};

		const showModal = () => {
			open.value = true;
			confirmLoading.value = false;
			console.log(responseData.value);
			if (responseData.value && responseData.value.account) {
				newUsername.value = responseData.value.account.username;
				imageUrl.value = responseData.value.account.avatar; // Fallback to empty string
				newGender.value = responseData.value.account.gender;
				newAge.value = responseData.value.account.age;
			} else {
				// Handle case where responseData is not yet loaded or malformed
				newUsername.value = '';
				imageUrl.value = '';
				newGender.value = undefined;
				newAge.value = undefined;
			}
			fileList.value = []; // Clear previous file list for a-upload
		};

		const handleModalCancel = () => {
			isUploading.value = false;
			open.value = false;
		};

		const handleOk = async () => {
			confirmLoading.value = true;

			let newAvatar = '';

			if (isUploading.value && cropper.value) {
				// 裁剪并上传
				console.log('正在裁剪');
				await new Promise((resolve) => {
					cropper.value.getCropData((croppedData) => {
						imageUrl.value = croppedData; // Update imageUrl with cropped data
						isUploading.value = false;
						resolve('裁剪完成');
					});
				});
				newAvatar = await uploadToGithub(imageUrl.value);
			} else {
				console.log('无需裁剪');
				newAvatar = imageUrl.value;
			}

			if (!newUsername.value || !newAge.value || !newAvatar || (newGender.value !== 0 && newGender.value !== 1)) {
				antd.message.error('请填写所有必填项');
				open.value = false;
				confirmLoading.value = false; // Reset loading state
				return;
			}

			const response = await axios.post('http://127.0.0.1:5000/api/updateInfo', {
				username: username,
				token: token,
				newUsername: newUsername.value,
				newAvatar: newAvatar,
				newGender: newGender.value,
				newAge: newAge.value,
			});

			if (response.status === 200) {
				antd.message.success('用户信息更新成功');
				await fetchInfo();
			} else {
				antd.message.error(`用户信息更新失败`);
			}

			open.value = false; // Close modal after handling
			confirmLoading.value = false; // Reset loading state
		}

		return {
			responseData,
			newUsername,
			newGender,
			newAge,
			open,
			confirmLoading,
			imageUrl,
			loading,
			fileList,
			isUploading,
			cropper,
			fetchInfo,
			beforeUpload,
			showModal,
			handleOk,
			handleModalCancel
		};
	}
});

export default ProfileInfo;