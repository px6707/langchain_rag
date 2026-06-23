import api from './index'
import { setAuth, type AuthUser } from '@/stores/auth'

export interface LoginRequest {
  username: string
  password: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  user: AuthUser
}

export async function login(username: string, password: string): Promise<AuthUser> {
  const { data } = await api.post<TokenResponse>('/auth/login', { username, password })
  setAuth(data.access_token, data.user)
  return data.user
}

export async function getMe(): Promise<AuthUser> {
  const { data } = await api.get<AuthUser>('/auth/me')
  return data
}
