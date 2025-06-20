<script lang="ts" setup>
  // import { ref } from 'vue';
  import AttackRecords from '@/pages/table/AttackRecords.vue';
  import ProfileInfo from './ProfileInfo.vue';

  // const select = ref('overview');
</script>
<template>
		<div class="mt-2 flex justify-evenly">
			<ProfileInfo class="flex-1 ml-lg mr-20" />
			<AttackRecords :default-current-page-size="3" :only-treasure="true"/>
			<div class="p-8"></div>
		</div>
</template>
