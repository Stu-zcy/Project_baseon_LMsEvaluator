<script lang="ts" setup>
import AttackRecords from '@/pages/table/AttackRecords.vue';
</script>

<template>
	<AttackRecords :defaultCurrentPageSize="5" :onlyTreasure="false"/>
</template>