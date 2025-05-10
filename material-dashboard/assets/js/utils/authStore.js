class AuthStore {
	constructor() {
		// 从 localStorage 获取存储的数据
		this.token = localStorage.getItem('Global_token');
		this.token_refresh = localStorage.getItem('Global_token_refresh');
		this.username = localStorage.getItem('Global_username');

		// // 尝试解析用户信息
		// try {
		// 		this.accountInfo = JSON.parse(localStorage.getItem('Global_accountInfo') || '{}');
		// } catch (e) {
		// 		this.accountInfo = {};
		// 		console.error('用户信息解析失败:', e);
		// }

		// 检查是否已登录
		this.isLogged = !!this.token && !!this.token_refresh;
	}

	// 设置登录信息
	setLoginInfo(data, username) {
		const { token, token_refresh } = data.data;

		// 更新内存中的数据
		this.token = token;
		this.token_refresh = token_refresh;
		this.username = username
		this.isLogged = true;

		// 保存到 localStorage
		localStorage.setItem('Global_token', token);
		localStorage.setItem('Global_token_refresh', token_refresh);
		localStorage.setItem('Global_username', this.username);
	}

	// 更新 token
	updateToken(newToken, newFreshToken) {
		this.token = newToken;
		this.token_refresh = newFreshToken;
		localStorage.setItem('Global_token', newToken);
		localStorage.setItem('Global_token_refresh', newFreshToken);
	}

	// 获取 token
	getToken() {
		return this.token;
	}

	// 获取刷新 token
	gettoken_refresh() {
		return this.token_refresh;
	}

	// 获取用户名
	getUsername() {
		return this.username;
	}


	// 清除登录信息
	clearLoginInfo() {
		this.token = null;
		this.token_refresh = null;
		this.username = null;
		this.accountInfo = {};
		this.isLogged = false;

		localStorage.removeItem('Global_token');
		localStorage.removeItem('Global_token_refresh');
		localStorage.removeItem('Global_username');
	}

	// 检查是否登录
	checkLogin() {
		return this.isLogged;
	}
}

// 创建单例
const authStore = new AuthStore();
export default authStore;