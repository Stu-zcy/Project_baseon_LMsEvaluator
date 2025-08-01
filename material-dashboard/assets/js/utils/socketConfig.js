import authStore from './authStore.js'; // 假设你的 authStore 存在且可用

const BACKEND_WS_URL = 'http://39.105.14.21:5000'; // 你的后端 Flask Socket.IO 地址

// 全局 socket 实例
let globalSocketInstance = null;

const createSocketInstance = () => {
  if (globalSocketInstance && globalSocketInstance.connected) {
    console.log('Socket instance already exists and is connected.');
    return globalSocketInstance;
  }

  // 获取认证 token
  const token = authStore.getToken();

  // 创建 Socket.IO 实例
  const socket = io(BACKEND_WS_URL, {
    transports: ['websocket', 'polling'], // 优先使用 WebSocket
    reconnection: true, // 允许自动重连
    reconnectionAttempts: 5, // 重连尝试次数
    reconnectionDelay: 1000, // 重连延迟 (毫秒)
    // 认证：在连接握手时发送 token
    auth: {
      token: token
    },
    // query: { // 也可以通过查询参数发送 token (不推荐用于敏感信息)
    //   token: token
    // }
  });

  // -----------------------------------------------------------
  // 全局事件监听 (所有使用这个 socket 实例的地方都会共享这些事件)
  // -----------------------------------------------------------

  socket.on('connect', () => {
    console.log('SocketService: Global WebSocket connected!');
    // 可以在这里重新发送认证信息，如果 token 刷新了
    // socket.emit('reauthenticate', { token: authStore.getToken() });
  });

  socket.on('disconnect', (reason) => {
    console.log('SocketService: Global WebSocket disconnected:', reason);
    // 根据断开原因处理，例如 token 过期等
    if (reason === 'io server disconnect' || reason === 'transport close' || reason === 'ping timeout') {
      // 如果是服务器强制断开，可能需要重新认证或提示用户登录
      console.warn('SocketService: Server-side disconnection, might need re-auth.');
      // 可以在这里触发重新登录流程或 token 刷新（如果 socketio 提供了刷新 token 的 hook）
      // if (authStore.getRefreshToken()) {
      //   // 尝试刷新 token 并重连
      //   authStore.refreshToken().then(() => {
      //     socket.connect(); // 重新连接
      //   }).catch(() => {
      //     // 刷新失败，提示用户重新登录
      //     antd.message.error('登录已过期，请重新登录');
      //     authStore.clearLoginInfo();
      //     window.location.href = '/pages/sign-in.html';
      //   });
      // }
    }
  });

  socket.on('connect_error', (error) => {
    console.error('SocketService: Global WebSocket connection error:', error.message);
    // if (error.message.includes('Authentication failed')) {
    //   // 如果是认证失败，可能需要清空token并重新登录
    //   antd.message.error('WebSocket 认证失败，请重新登录');
    //   authStore.clearLoginInfo();
    //   window.location.href = '/pages/sign-in.html';
    // }
  });

  // 你可以添加更多全局事件监听器，例如错误信息、通用通知等
  // socket.on('error_message', (msg) => { console.error('Socket Error:', msg); });

  globalSocketInstance = socket;
  return globalSocketInstance;
};

// 在需要的时候手动断开连接 (例如用户登出时)
const disconnectSocket = () => {
  if (globalSocketInstance && globalSocketInstance.connected) {
    globalSocketInstance.disconnect();
    globalSocketInstance = null;
    console.log('SocketService: Global WebSocket disconnected explicitly.');
  }
};

const getSocket = () => {
  return globalSocketInstance;
};

export default {
  createSocketInstance,
  getSocket,
  disconnectSocket
};