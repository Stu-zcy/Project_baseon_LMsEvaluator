<template>
  <a-layout class="layout">
    <a-layout-content class="content">
      <a-card v-if="hasAccess" title="用户管理" class="management-card">
        <!-- 搜索区域 -->
        <div class="search-container">
          <a-input
            v-model:value="searchQuery"
            placeholder="搜索用户名"
            allow-clear
            @change="filterUsers"
            class="search-input"
          >
            <template #prefix>
              <search-outlined />
            </template>
          </a-input>
        </div>

        <!-- 用户列表 -->
        <div class="user-list">
          <div v-for="user in pagedUsers" :key="user.username" class="user-card">
            <!-- 用户信息展示 -->
            <div class="user-info">
              <div class="avatar-wrapper">
                <img 
                  :src="user.avatar_url || 'default-avatar.png'" 
                  class="user-avatar"
                  @error="handleAvatarError"
                />
              </div>
              <div class="user-details">
                <h3>{{ user.username }}</h3>
                <div class="meta-info">
                  <span>年龄: {{ user.age }}</span>
                  <span>性别: {{ genderMap[user.gender] }}</span>
                  <span class="email">{{ user.email }}</span>
                </div>
              </div>
            </div>
            
            <!-- 操作按钮 -->
            <div class="action-buttons">
              <a-popconfirm
                title="确定要删除该用户吗？"
                @confirm="deleteUser(user.username)"
                ok-text="确认"
                cancel-text="取消"
              >
                <a-button type="link" danger>
                  <template #icon><delete-outlined /></template>
                </a-button>
              </a-popconfirm>
            </div>
          </div>
        </div>

        <!-- 分页控件 -->
        <a-pagination
          v-model:current="pagination.current"
          v-model:pageSize="pagination.pageSize"
          :total="pagination.total"
          show-size-changer
          class="pagination"
        />
      </a-card>

      <!-- 无权限访问提示 -->
      <div v-else class="no-access">
        <h2>无权限访问</h2>
      </div>
    </a-layout-content>
  </a-layout>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import { message } from 'ant-design-vue';
import { SearchOutlined, DeleteOutlined } from '@ant-design/icons-vue';
import axios from 'axios';

// 响应式数据
const searchQuery = ref('');
const pagination = ref({
  current: 1,
  pageSize: 5,
  total: 0,
});
const users = ref([]);
const genderMap = ref(['男', '女']);
const hasAccess = ref(true); // 用来控制是否有权限访问用户列表

// 计算属性
const filteredUsers = computed(() => {
  const query = searchQuery.value.toLowerCase();
  return users.value.filter(user => 
    user.username.toLowerCase().includes(query)
  );
});

const pagedUsers = computed(() => {
  const start = (pagination.value.current - 1) * pagination.value.pageSize;
  const end = start + pagination.value.pageSize;
  return filteredUsers.value.slice(start, end);
});

// 验证函数
const auth = async () => {
  const username = localStorage.getItem('Global_username');
  const token = localStorage.getItem('Global_token');

  try {
    const response = await axios.post('http://127.0.0.1:5000/api/auth', { username, token });

    if (response.data.status === 'success') {
      hasAccess.value = true;
      fetchUsers(); // 认证成功后获取用户数据
    } else {
      hasAccess.value = false; // 认证失败，显示无权限访问
    }
  } catch (error) {
    hasAccess.value = false; // 如果请求失败，也显示无权限访问
    message.error('认证失败');
  }
};

// 获取用户列表
const fetchUsers = async () => {
  try {
    const response = await axios.post('http://127.0.0.1:5000/api/users', {
      username: localStorage.getItem('Global_username'),
      token: localStorage.getItem('Global_token')
    });

    if (response.data.status === 'success') {
      users.value = response.data.users;
      pagination.value.total = response.data.users.length;
    }
  } catch (error) {
    message.error('获取用户列表失败');
  }
};

const deleteUser = async (username) => {
  try {
    const response = await axios.post('http://127.0.0.1:5000/api/delete_users', {
      username: localStorage.getItem('Global_username'),
      token: localStorage.getItem('Global_token'),
      delete_username: username
    });

    if (response.data.status === 'success') {
      message.success('删除成功');
      await fetchUsers();
    }
  } catch (error) {
    message.error('删除操作失败');
  }
};

const handleAvatarError = (e) => {
  e.target.src = 'default-avatar.png';
};

const filterUsers = () => {
  pagination.value.current = 1; // 重置分页到第一页
};

// 生命周期
onMounted(() => {
  auth(); // 在组件加载时调用认证函数
});
</script>

<style scoped>
.management-card {
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.user-list {
  margin-top: 24px;
}

.user-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  margin-bottom: 12px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  transition: transform 0.2s;
}

.user-card:hover {
  transform: translateY(-2px);
}

.user-info {
  display: flex;
  align-items: center;
  gap: 16px;
}

.avatar-wrapper {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  overflow: hidden;
  flex-shrink: 0;
}

.user-avatar {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.user-details h3 {
  margin: 0;
  color: #333;
}

.meta-info {
  display: flex;
  gap: 12px;
  color: #666;
  font-size: 0.9em;
}

.email {
  color: #1890ff;
}

.action-buttons {
  flex-shrink: 0;
}

.pagination {
  margin-top: 24px;
  text-align: center;
}

.search-container {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 24px;
}

.search-input {
  width: 300px;
}

.no-access {
  text-align: center;
  margin-top: 50px;
  font-size: 24px;
  color: red;
}
</style>
