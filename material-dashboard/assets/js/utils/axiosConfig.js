import axios from '../../../node_modules/axios/dist/esm/axios.js';
import authStore from './authStore.js';

// 创建 axios 实例
const instance = axios.create({
    baseURL: 'http://localhost:5000',
    timeout: 5000
});

// 是否正在刷新token
let isRefreshing = false;
// 请求队列
let requests = [];

// 请求拦截器
instance.interceptors.request.use(
    config => {
        const token = authStore.getToken();
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    error => Promise.reject(error)
);

// 响应拦截器
instance.interceptors.response.use(
    response => response,
    async error => {
        const originalRequest = error.config;

        // token相关错误处理
        if (error.response?.data.code === 403 && !originalRequest._retry) {
            if (isRefreshing) {
                // 将请求加入队列
                return new Promise(resolve => {
                    requests.push(() => {
                        resolve(instance(originalRequest));
                    });
                });
            }

            originalRequest._retry = true;
            isRefreshing = true;

            try {
                // 尝试刷新token
                const response = await instance.post('/api/refresh_token', {
                    refresh_token: authStore.getRefreshToken(),
                    username: authStore.getUsername()
                });
								
								console.log("刷新token：", response);
                if (response.data.code === 200) {
                    // 更新token
                    authStore.updateToken(response.data.data.token, response.data.data.token_refresh);
                    
                    // 执行队列中的请求
                    requests.forEach(callback => callback());
                    // requests = [];
                    
                    // 重试原始请求
                    return instance(originalRequest);
                }
            } catch (refreshError) {
                // 刷新token失败，清除用户信息并跳转到登录页
                authStore.clearLoginInfo();
                window.location.href = '/pages/sign-in.html';
                return Promise.reject(refreshError);
            } finally {
                isRefreshing = false;
								requests = [];
            }
        }

        // 其他错误处理
        switch (error.response?.data.code) {
            // case 400:
            //     alert('请求参数错误');
            //     break;
            // case 401:
            //     authStore.clearLoginInfo();
            //     window.location.href = '/pages/sign-in.html';
            //     break;
            // case 404:
            //     alert('资源不存在');
            //     break;
            default:
                console.error('请求失败:', error);
        }

        return Promise.reject(error);
    }
);

export default instance;