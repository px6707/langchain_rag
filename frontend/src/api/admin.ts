import api from './index'
import type { AuthUser } from '@/stores/auth'

export interface UserListResponse {
  items: AuthUser[]
  total: number
}

export interface UserCreateRequest {
  username: string
  password: string
  is_admin?: boolean
}

export interface UserUpdateRequest {
  is_active?: boolean
  is_admin?: boolean
  password?: string
}

export async function listUsers(): Promise<UserListResponse> {
  const { data } = await api.get<UserListResponse>('/admin/users')
  return data
}

export async function createUser(payload: UserCreateRequest): Promise<AuthUser> {
  const { data } = await api.post<AuthUser>('/admin/users', payload)
  return data
}

export async function updateUser(id: string, payload: UserUpdateRequest): Promise<AuthUser> {
  const { data } = await api.patch<AuthUser>(`/admin/users/${id}`, payload)
  return data
}

export async function deleteUser(id: string): Promise<void> {
  await api.delete(`/admin/users/${id}`)
}
