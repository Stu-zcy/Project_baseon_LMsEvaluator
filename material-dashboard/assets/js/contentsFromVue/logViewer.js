const { defineComponent, ref, onMounted, onUnmounted, nextTick } = Vue;

import socketService from '../utils/socketConfig.js';


export default defineComponent({
  name: 'logViewer',

  props: {
    username: {
      type: String,
      required: true
    },
    targetCreateTime: {
      type: Number,
      required: true
    }
  },

  setup(props) {
    const socket = ref(null);

    const logs = ref('');
    const connectionStatus = ref('disconnected');
    const logDisplay = ref(null);

    const initializeSocket = () => {
      // 获取或创建 Socket 实例
      socket.value = socketService.createSocketInstance();

      // **只在 LogViewer 组件内部监听它特有的事件**
      // 全局的 connect/disconnect/connect_error 事件已经在 socketService 中处理，
      // LogViewer 只需要根据 socket 状态来更新自己的 UI。
      // 如果 LogViewer 确实需要自己对这些全局事件作出特定反应，可以继续监听。
      
      // 监听 socket 的状态变化以更新 UI
      socket.value.on('connect', () => {
        connectionStatus.value = 'connected';
        // antd.message.success('LogViewer: WebSocket 连接成功！'); // 避免重复提示
        console.log('LogViewer: Connected to WebSocket server.');
        requestLogStream(); // 连接成功后立即请求日志流
      });

      socket.value.on('disconnect', (reason) => {
        connectionStatus.value = 'disconnected';
        // antd.message.warning('LogViewer: WebSocket 连接断开: ' + reason); // 避免重复提示
        console.log('LogViewer: Disconnected from WebSocket server:', reason);
        logs.value += `\n--- WebSocket 连接断开: ${reason} --- \n`;
      });

      socket.value.on('connect_error', (error) => {
        connectionStatus.value = 'disconnected';
        // antd.message.error('LogViewer: WebSocket 连接错误: ' + error.message); // 避免重复提示
        console.error('LogViewer: WebSocket connection error:', error);
        logs.value += `\n--- WebSocket 连接错误: ${error.message} --- \n`;
      });

      // 监听后端推送的日志内容事件 (LogViewer 特有)
      socket.value.on('new_log_entry', (data) => {
        logs.value += data;
        nextTick(() => {
          if (logDisplay.value) {
            logDisplay.value.scrollTop = logDisplay.value.scrollHeight;
          }
        });
      });

      // 监听后端发送的状态/通知事件 (LogViewer 特有)
      socket.value.on('status', (data) => {
        console.log('LogViewer: Server status:', data.data);
      });

      // 根据当前 socket 连接状态设置初始状态
      connectionStatus.value = socket.value.connected ? 'connected' : 'connecting';
    };


    const requestLogStream = () => {
      if (!socket.value || !socket.value.connected) {
        console.warn('LogViewer: Socket 未连接，无法发送 requestLog 请求。等待连接成功或手动重试。');
        return;
      }
			logs.value = ''
			console.log('即将监听日志，用户名：', props.username)
      const payload = {
        username: props.username,
        createTime: props.targetCreateTime
      };
      socket.value.emit('requestLog', payload);
      console.log('LogViewer: 发出 requestLog 事件， payload:', payload);
    };

    const stopLogStream = () => {
      if (socket.value && socket.value.connected) {
        socket.value.emit('stopLog');
        console.log('LogViewer: 发出 stopLog 事件');
      }
    };

    // -------------------------------------------------------------------
    // 生命周期钩子
    // -------------------------------------------------------------------

    onMounted(() => {
      // 组件挂载时初始化 Socket
      initializeSocket();
    });

    onUnmounted(() => {
      // 组件卸载时，通知后端停止日志流
      stopLogStream();
      
      // **重要：移除 LogViewer 自身添加的监听器，但不要断开全局连接**
      // 因为 socketService 已经管理了连接的生命周期
      if (socket.value) {
          socket.value.off('connect');
          socket.value.off('disconnect');
          socket.value.off('connect_error');
          socket.value.off('new_log_entry');
          socket.value.off('status');
      }
			socketService.disconnectSocket();
      console.log('LogViewer: Global Socket disconnected on component unmount.');
    });

    return {
      logs,
      connectionStatus,
      logDisplay
    };
  },

  template: `
    <div class="log-viewer-container">
      <p v-if="connectionStatus === 'connecting'">正在连接 WebSocket...</p>
      <p v-else-if="connectionStatus === 'connected'" style="color: green;">正在与后端保持通信</p>
      <p v-else style="color: red;">WebSocket 已断开或连接失败</p>

      <pre ref="logDisplay" class="log-display">{{ logs }}</pre>
    </div>
  `
});