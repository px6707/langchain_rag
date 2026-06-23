const TOKEN_KEY = 'access_token'
const USER_KEY = 'user'

export interface AuthUser {
  id: string
  username: string
  is_admin: boolean
  is_active: boolean
  created_at: string
}

interface AuthState {
  accessToken: string | null
  user: AuthUser | null
}

const state: AuthState = {
  accessToken: localStorage.getItem(TOKEN_KEY),
  user: loadUser(),
}

function loadUser(): AuthUser | null {
  const raw = localStorage.getItem(USER_KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw) as AuthUser
  } catch {
    return null
  }
}

export function getAccessToken(): string | null {
  return state.accessToken
}

export function getUser(): AuthUser | null {
  return state.user
}

export function isAuthenticated(): boolean {
  return !!state.accessToken && !!state.user
}

export function isAdmin(): boolean {
  return !!state.user?.is_admin
}

export function setAuth(token: string, user: AuthUser): void {
  state.accessToken = token
  state.user = user
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(USER_KEY, JSON.stringify(user))
}

export function clearAuth(): void {
  state.accessToken = null
  state.user = null
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

export function logout(): void {
  clearAuth()
}
