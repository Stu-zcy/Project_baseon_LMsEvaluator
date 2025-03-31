<script lang="ts" setup>
import axios from 'axios';
import { ref, onMounted } from 'vue';
import { message } from 'ant-design-vue';

const username = localStorage.getItem('Global_username');
const token = localStorage.getItem('Global_token');
const responseData = ref<any>({});
const newUsername = ref<string>();
const newGender = ref<number>();
const newAge = ref<number>();
const open = ref<boolean>(false);
const confirmLoading = ref<boolean>(false);
const imageUrl = ref<string>('');  // 预览图像的URL
const loading = ref<boolean>(false);
const fileList = ref<object[]>([]);

// 裁剪框显示控制
const cropperVisible = ref<boolean>(false);
const cropperTitle = ref<string>('裁剪头像');
const canvasRef = ref<HTMLCanvasElement | null>(null);
const ctx = ref<CanvasRenderingContext2D | null>(null);
const img = ref<HTMLImageElement | null>(null);
const cropArea = ref<{ x: number; y: number; width: number; height: number }>({
  x: 50,
  y: 50,
  width: 200,
  height: 200,
});

// 新增标志，判断图像是否已经裁剪
const isCropped = ref<boolean>(false);

const isDragging = ref<boolean>(false);
const lastX = ref<number>(0);
const lastY = ref<number>(0);

// 获取用户信息
async function fetchInfo() {
  try {
    const response = await axios.post('http://127.0.0.1:5000/api/profile', {
      username: username,
      token: token,
    });
    responseData.value = response.data.data;
  } catch (error) {
    console.error('请求出错', error);
  }
}

onMounted(async () => {
  await fetchInfo();
});

// 头像上传前检查
const beforeUpload = (file) => {
  loading.value = true;
  const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
  const isLt2M = file.size / 1024 / 1024 < 20;
  if (!isJpgOrPng) {
    message.error('只能上传 JPG 或 PNG 格式的图片');
    loading.value = false;
  } else if (!isLt2M) {
    message.error('图片大小必须小于 20MB');
    loading.value = false;
  } else {
    // 预览
    const reader = new FileReader();
    reader.onload = (e) => {
      imageUrl.value = e.target.result as string;
      cropperVisible.value = true;  // Show the cropping interface
      initCrop(); // 初始化裁剪区域
    };
    reader.readAsDataURL(file);
  }
  return false;
};

// 初始化裁剪区域
const initCrop = () => {
  if (canvasRef.value && imageUrl.value) {
    const canvas = canvasRef.value;
    ctx.value = canvas.getContext('2d');
    img.value = new Image();
    img.value.onload = () => {
      canvas.width = img.value.width;
      canvas.height = img.value.height;
      ctx.value?.drawImage(img.value, 0, 0);
      drawCropArea();
    };
    img.value.src = imageUrl.value;
  }
};

// 绘制裁剪框
const drawCropArea = () => {
  if (ctx.value && img.value) {
    // 绘制图片
    ctx.value.clearRect(0, 0, canvasRef.value!.width, canvasRef.value!.height);
    ctx.value.drawImage(img.value, 0, 0);

    // 绘制裁剪框
    ctx.value.beginPath();
    ctx.value.rect(cropArea.value.x, cropArea.value.y, cropArea.value.width, cropArea.value.height);
    ctx.value.lineWidth = 2;
    ctx.value.strokeStyle = '#FF6347'; // 边框颜色
    ctx.value.setLineDash([6, 6]); // 虚线效果
    ctx.value.stroke();
    ctx.value.closePath();
  }
};

// 启动裁剪框拖动
const startDragging = (event: MouseEvent) => {
  isDragging.value = true;
  lastX.value = event.clientX;
  lastY.value = event.clientY;
};

// 更新裁剪框位置
const dragCropArea = (event: MouseEvent) => {
  if (isDragging.value) {
    const dx = event.clientX - lastX.value;
    const dy = event.clientY - lastY.value;
    cropArea.value.x += dx;
    cropArea.value.y += dy;
    lastX.value = event.clientX;
    lastY.value = event.clientY;
    drawCropArea();
  }
};

// 结束裁剪框拖动
const stopDragging = () => {
  isDragging.value = false;
};

// 放大裁剪框
const zoomInCropArea = () => {
  cropArea.value.width += 20;
  cropArea.value.height += 20;
  drawCropArea();
};

// 缩小裁剪框
const zoomOutCropArea = () => {
  if (cropArea.value.width > 40 && cropArea.value.height > 40) {
    cropArea.value.width -= 20;
    cropArea.value.height -= 20;
    drawCropArea();
  }
};

// 确认裁剪后的图像并上传
const cropperImgMethod = () => {
  if (canvasRef.value && ctx.value && img.value) {
    // 创建一个新的canvas来存储裁剪后的图像
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    // 设置裁剪后的canvas大小
    canvas.width = cropArea.value.width;
    canvas.height = cropArea.value.height;
    
    // 使用裁剪框的坐标和大小从原始图像中提取裁剪区域
    context?.drawImage(
      img.value,
      cropArea.value.x, 
      cropArea.value.y, 
      cropArea.value.width, 
      cropArea.value.height, 
      0, 
      0, 
      cropArea.value.width, 
      cropArea.value.height
    );
    
    // 获取裁剪后的图像的Base64格式
    const croppedImg = canvas.toDataURL('image/png');
    imageUrl.value = croppedImg; // 更新为裁剪后的图像
    isCropped.value = true; // 设置裁剪标志为true
    
    // 调用上传裁剪后的头像函数
    uploadAvatar(croppedImg); 

    cropperVisible.value = false; // 关闭裁剪框
  }
};

// 上传裁剪后的图像
const uploadAvatar = async (file: string) => {
  const currentTime = new Date().toISOString().replace(/[:.]/g, '-');
  const fileName = currentTime + '.png'; // 仅处理裁剪后的图像
  const rawUrl = 'https://api.github.com/repos/Stu-zcy/Public_image/contents/';
  const fastUrl = 'https://raw.githubusercontent.com/Stu-zcy/Public_image/refs/heads/main/';
  const token = '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';

  // 只上传裁剪后的图像
  const base64Img = file.split(',')[1];
  const response = await fetch(rawUrl + fileName, {
    method: 'PUT',
    headers: {
      'Authorization': 'token ' + token,
      'Accept': 'application/vnd.github.v3+json',
    },
    body: JSON.stringify({
      message: 'Upload cropped avatar',
      content: base64Img,
      branch: 'main',
    }),
  });

  
  if (response.status === 201) {
    console.log('裁剪后的图片上传成功');
    return fastUrl + fileName; // 返回上传后的图像链接
  } else {
    console.error('上传失败');
    return '';
  }
};

// 关闭裁剪框
const closeCropperDialog = () => {
  cropperVisible.value = false;
};

// 点击修改按钮时弹出模态框
const showModal = () => {
  open.value = true;
  confirmLoading.value = false;
  newUsername.value = responseData.value['account']['username'];
  imageUrl.value = responseData.value['account']['avatar'];
  newGender.value = responseData.value['account']['gender'];
  newAge.value = responseData.value['account']['age'];
};

// 确认修改
const handleOk = async () => {
  confirmLoading.value = true;
  const newAvatar = await uploadAvatar(imageUrl.value);  // 只上传裁剪后的图像
  const response = await axios.post('http://127.0.0.1:5000/api/updateInfo', {
    username: username,
    token: token,
    newUsername: newUsername.value,
    newAvatar: newAvatar,
    newGender: newGender.value,
    newAge: newAge.value,
  });
  await fetchInfo();
  open.value = false;
  confirmLoading.value = false;
};

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
          <span class="label w-5">用户名：</span>
          <span>{{ responseData['account']['username'] }}</span>
        </li>
        <li class="infoItem">
          <span class="label w-5">权限等级：</span>
          <span>{{ responseData['role'] }}</span>
        </li>
        <li class="infoItem">
          <span class="label w-5">年龄：</span>
          <span>{{ responseData['account']['age'] }}</span>
        </li>
        <li class="infoItem">
          <span class="label w-2">性别：</span>
          <span>{{ responseData['account']['gender'] === 0 ? '女' : '男' }}</span>
        </li>
      </ul>
      <template #actions>
        <edit-outlined key="edit" @click="showModal" />
      </template>
    </a-card>

    <!-- 修改信息弹窗 -->
    <a-modal v-if="open" :visible="open" title="修改信息" @ok="handleOk" :confirm-loading="confirmLoading" :closable="false" @cancel="() => { open.value = false }" :centered="true" width="350px">
      <a-form :label-col="{ style: { width: '100px' } }" class="w-full mx-auto">
        <a-form-item label="头像">
          <a-upload v-model:file-list="fileList" name="avatar" list-type="picture-card" :show-upload-list="false" maxCount="1" :before-upload="beforeUpload">
            <img :src="imageUrl" class="w-full h-full" />
          </a-upload>
          <canvas
            ref="canvasRef"
            class="w-full h-full"
            @mousedown="startDragging"
            @mousemove="resizeCropArea"
            @mouseup="stopResizing"
            @mouseleave="stopResizing"
          ></canvas>
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
        <a-form-item label="年龄">
          <a-input-number v-model:value="newAge" :min="0" :max="99"></a-input-number>
        </a-form-item>
        <a-form-item>
          <a-button @click="cropperImgMethod" type="primary">确认裁剪</a-button>
          <a-button @click="zoomInCropArea" type="default" class="ml-2">放大</a-button>
          <a-button @click="zoomOutCropArea" type="default" class="ml-2">缩小</a-button>
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<style scoped>
canvas {
  max-width: 90%;
  max-height: 90%;
  border: 1px solid #ccc;
  margin-top: 20px;
  cursor: pointer;
}
</style>
