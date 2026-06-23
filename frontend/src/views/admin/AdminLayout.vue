<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white border-b border-gray-200 shadow-sm">
      <div class="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <el-icon :size="24" class="text-blue-600"><Setting /></el-icon>
          <span class="text-lg font-semibold text-gray-800">管理中心</span>
        </div>
        <div class="flex items-center gap-3">
          <span class="text-sm text-gray-600">{{ user?.username }}</span>
          <router-link to="/">
            <el-button text>返回业务区</el-button>
          </router-link>
          <el-button text type="danger" @click="handleLogout">退出</el-button>
        </div>
      </div>
    </header>

    <div class="flex-1 max-w-6xl w-full mx-auto px-4 py-6 flex gap-6">
      <aside class="w-48 shrink-0">
        <el-menu :default-active="activeMenu" router class="rounded-lg border border-gray-200">
          <el-menu-item index="/admin/users">
            <el-icon><User /></el-icon>
            <span>用户维护</span>
          </el-menu-item>
        </el-menu>
      </aside>
      <main class="flex-1 min-w-0">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Setting, User } from '@element-plus/icons-vue'
import { getUser, logout } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const user = computed(() => getUser())
const activeMenu = computed(() => route.path)

function handleLogout() {
  logout()
  router.replace('/login')
}
</script>
