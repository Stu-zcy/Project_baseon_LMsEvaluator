import { defineStore } from 'pinia';
import http from './http';
import { Response } from '@/types';
import { useMenuStore } from './menu';
import { useAuthStore } from '@/plugins';
import { useLoadingStore } from './loading';

export interface Profile {
  account: Account;
  permissions: string[];
  role: string;
}
export interface Account {
  username: string;
  avatar: string;
  gender: number;
}

export type TokenResult = {
  token: string;
  expires: number;
};
export const useAccountStore = defineStore('account', {
  state() {
    return {
      account: {} as Account,
      permissions: [] as string[],
      role: '',
      logged: true,
    };
  },
  actions: {
    async login(username: string, password: string) {
      return http
        .request<TokenResult, Response<TokenResult>>('http://localhost:5000/api/login', 'post_json', { username, password })
        .then(async (response) => {
          if (response.code === 0) {
            // 登录成功，更新登录状态
            this.logged = true;
            const { token, expires } = response.data;
            http.setAuthorization(`Bearer ${token}`, new Date(expires));
    
            // 从响应中提取账户信息并存储到 localStorage
            const { accountInfo } = response;
            
            console.log('accountInfo', accountInfo);
            this.account = accountInfo.account;
            this.permissions = accountInfo.permissions;
            this.role = accountInfo.role;
            localStorage.setItem('Global_username', JSON.stringify(username));  // 存储用户名
            localStorage.setItem('Global_token', JSON.stringify(token)); 
            // 获取菜单
            await useMenuStore().getMenuList();
    
            return response.data;
          } else {
            return Promise.reject(response);
          }
        });
    },    
    async logout() {
      return new Promise<boolean>((resolve) => {
        localStorage.removeItem('stepin-menu');
        http.removeAuthorization();
        this.logged = false;
        resolve(true);
      });
    },
    async profile() {
      const { setAuthLoading } = useLoadingStore();
      setAuthLoading(true);
      return http
        .request<Account, Response<Profile>>('/account', 'get')
        .then((response) => {
          if (response.code === 0) {
            const { setAuthorities } = useAuthStore();
            const { account, permissions, role } = response.data;
            this.account = account;
            this.permissions = permissions;
            this.role = role;
            setAuthorities(permissions);
            return response.data;
          } else {
            return Promise.reject(response);
          }
        })
        .finally(() => setAuthLoading(false));
    },
    setLogged(logged: boolean) {
      this.logged = logged;
    },
  },
});
