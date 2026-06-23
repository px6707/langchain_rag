import { createRouter, createWebHistory } from 'vue-router'
import { getAccessToken, isAdmin, setAuth } from '@/stores/auth'
import { getMe } from '@/api/auth'
import ChatView from '@/views/ChatView.vue'
import UploadView from '@/views/UploadView.vue'
import LoginView from '@/views/LoginView.vue'
import AdminLayout from '@/views/admin/AdminLayout.vue'
import UsersView from '@/views/admin/UsersView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: LoginView,
      meta: { guest: true },
    },
    {
      path: '/',
      name: 'chat',
      component: ChatView,
      meta: { requiresAuth: true },
    },
    {
      path: '/upload',
      name: 'upload',
      component: UploadView,
      meta: { requiresAuth: true },
    },
    {
      path: '/admin',
      component: AdminLayout,
      meta: { requiresAuth: true, requiresAdmin: true },
      children: [
        {
          path: '',
          redirect: '/admin/users',
        },
        {
          path: 'users',
          name: 'admin-users',
          component: UsersView,
          meta: { requiresAuth: true, requiresAdmin: true },
        },
      ],
    },
  ],
})

let authChecked = false

router.beforeEach(async (to) => {
  if (to.meta.guest) {
    if (getAccessToken()) {
      return '/'
    }
    return true
  }

  if (to.meta.requiresAuth) {
    const token = getAccessToken()
    if (!token) {
      return { path: '/login', query: { redirect: to.fullPath } }
    }

    if (!authChecked) {
      try {
        const user = await getMe()
        setAuth(token, user)
        authChecked = true
      } catch {
        return { path: '/login', query: { redirect: to.fullPath } }
      }
    }

    if (to.meta.requiresAdmin && !isAdmin()) {
      return '/'
    }
  }

  return true
})

export default router
