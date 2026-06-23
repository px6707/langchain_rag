<template>
  <header class="bg-white border-b border-gray-200 shadow-sm">
    <div class="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <el-icon :size="24" class="text-blue-600"><ChatDotRound /></el-icon>
        <span class="text-lg font-semibold text-gray-800">LangChain RAG</span>
      </div>
      <div class="flex items-center gap-4">
        <nav class="flex gap-2">
          <router-link to="/">
            <el-button :type="route.path === '/' ? 'primary' : 'default'" text>
              <el-icon class="mr-1"><ChatLineRound /></el-icon>
              对话
            </el-button>
          </router-link>
          <router-link to="/upload">
            <el-button :type="route.path === '/upload' ? 'primary' : 'default'" text>
              <el-icon class="mr-1"><Upload /></el-icon>
              文档上传
            </el-button>
          </router-link>
        </nav>
        <div class="flex items-center gap-2 border-l border-gray-200 pl-4">
          <span class="text-sm text-gray-600">{{ user?.username }}</span>
          <router-link v-if="user?.is_admin" to="/admin/users">
            <el-button text>
              <el-icon class="mr-1"><Setting /></el-icon>
              管理中心
            </el-button>
          </router-link>
          <el-button text type="danger" @click="handleLogout">退出</el-button>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ChatDotRound, ChatLineRound, Setting, Upload } from '@element-plus/icons-vue'
import { getUser, logout } from '@/stores/auth'

const route = useRoute()
const router = useRouter()
const user = computed(() => getUser())

function handleLogout() {
  logout()
  router.replace('/login')
}
</script>
