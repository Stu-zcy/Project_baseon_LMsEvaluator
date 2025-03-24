<template>
    <a-modal
      v-if="dialogVisible"
      :visible="dialogVisible"
      :title="title"
      :maskClosable="false"
      @cancel="closeCropperDialog"
      @ok="cropImage"
      :centered="true"
      :width="400"
    >
      <div class="cropper-container">
        <vue-cropper
          ref="cropper"
          :src="imageUrl"
          :aspect-ratio="1"
          :preview="previewSelector"
          :guides="false"
          :background="true"
          :auto-crop="true"
          :rotatable="false"
          :scalable="true"
          :zoomable="true"
        />
      </div>
    </a-modal>
  </template>
  
  <script lang="ts">
  import { ref, defineComponent } from 'vue';
  import VueCropper from 'vue-cropperjs';
  import 'cropperjs/dist/cropper.css';
  
  export default defineComponent({
    name: 'ImgCropper',
    components: {
      VueCropper,
    },
    props: {
      dialogVisible: {
        type: Boolean,
        required: true,
      },
      imageUrl: {
        type: String,
        required: true,
      },
      title: {
        type: String,
        default: '裁剪头像',
      },
      previewSelector: {
        type: String,
        default: '.img-preview',
      },
    },
    emits: ['closeCropperDialog', 'cropperImgMethod'],
    setup(props, { emit }) {
      const cropperRef = ref(null);
  
      const cropImage = () => {
        const cropper = cropperRef.value?.cropper;
        if (cropper) {
          const croppedImg = cropper.getCroppedCanvas().toDataURL();
          emit('cropperImgMethod', croppedImg);
        }
      };
  
      const closeCropperDialog = () => {
        emit('closeCropperDialog');
      };
  
      return {
        cropperRef,
        cropImage,
        closeCropperDialog,
      };
    },
  });
  </script>
  
  <style scoped>
  .cropper-container {
    max-width: 100%;
    height: 400px;
  }
  </style>
  