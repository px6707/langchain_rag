<template>
  <el-card>
    <template #header>
      <div class="flex items-center justify-between">
        <span class="font-medium">用户列表</span>
        <div class="flex gap-2">
          <el-button :icon="Refresh" circle size="small" @click="fetchUsers" />
          <el-button type="primary" :icon="Plus" @click="showCreateDialog = true">
            添加用户
          </el-button>
        </div>
      </div>
    </template>

    <el-table v-loading="loading" :data="users" stripe style="width: 100%">
      <el-table-column prop="username" label="用户名" min-width="140" />
      <el-table-column label="角色" width="100">
        <template #default="{ row }">
          <el-tag :type="row.is_admin ? 'warning' : 'info'" size="small">
            {{ row.is_admin ? '管理员' : '普通用户' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="row.is_active ? 'success' : 'danger'" size="small">
            {{ row.is_active ? '启用' : '禁用' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column label="创建时间" width="180">
        <template #default="{ row }">
          {{ formatDate(row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="260" fixed="right">
        <template #default="{ row }">
          <el-button
            size="small"
            :type="row.is_active ? 'warning' : 'success'"
            text
            @click="toggleActive(row)"
          >
            {{ row.is_active ? '禁用' : '启用' }}
          </el-button>
          <el-button size="small" text @click="openResetPassword(row)">重置密码</el-button>
          <el-popconfirm
            title="确定删除此用户？"
            :disabled="row.id === currentUserId"
            @confirm="handleDelete(row.id)"
          >
            <template #reference>
              <el-button
                size="small"
                type="danger"
                text
                :disabled="row.id === currentUserId"
              >
                删除
              </el-button>
            </template>
          </el-popconfirm>
        </template>
      </el-table-column>
    </el-table>
  </el-card>

  <el-dialog v-model="showCreateDialog" title="添加用户" width="420px" @closed="resetCreateForm">
    <el-form ref="createFormRef" :model="createForm" :rules="createRules" label-width="90px">
      <el-form-item label="用户名" prop="username">
        <el-input v-model="createForm.username" placeholder="登录用户名" />
      </el-form-item>
      <el-form-item label="密码" prop="password">
        <el-input v-model="createForm.password" type="password" show-password placeholder="至少 6 位" />
      </el-form-item>
      <el-form-item label="管理员">
        <el-switch v-model="createForm.is_admin" />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="showCreateDialog = false">取消</el-button>
      <el-button type="primary" :loading="submitting" @click="handleCreate">确定</el-button>
    </template>
  </el-dialog>

  <el-dialog v-model="showResetDialog" title="重置密码" width="420px" @closed="resetPasswordForm">
    <el-form ref="resetFormRef" :model="resetForm" :rules="resetRules" label-width="90px">
      <el-form-item label="用户">
        <span>{{ resetTarget?.username }}</span>
      </el-form-item>
      <el-form-item label="新密码" prop="password">
        <el-input v-model="resetForm.password" type="password" show-password placeholder="至少 6 位" />
      </el-form-item>
    </el-form>
    <template #footer>
      <el-button @click="showResetDialog = false">取消</el-button>
      <el-button type="primary" :loading="submitting" @click="handleResetPassword">确定</el-button>
    </template>
  </el-dialog>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage, type FormInstance, type FormRules } from 'element-plus'
import { Plus, Refresh } from '@element-plus/icons-vue'
import { createUser, deleteUser, listUsers, updateUser } from '@/api/admin'
import type { AuthUser } from '@/stores/auth'
import { getUser } from '@/stores/auth'

const loading = ref(false)
const submitting = ref(false)
const users = ref<AuthUser[]>([])
const showCreateDialog = ref(false)
const showResetDialog = ref(false)
const resetTarget = ref<AuthUser | null>(null)

const createFormRef = ref<FormInstance>()
const resetFormRef = ref<FormInstance>()

const createForm = reactive({
  username: '',
  password: '',
  is_admin: false,
})

const resetForm = reactive({
  password: '',
})

const createRules: FormRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码至少 6 位', trigger: 'blur' },
  ],
}

const resetRules: FormRules = {
  password: [
    { required: true, message: '请输入新密码', trigger: 'blur' },
    { min: 6, message: '密码至少 6 位', trigger: 'blur' },
  ],
}

const currentUserId = computed(() => getUser()?.id ?? '')

function formatDate(value: string): string {
  return new Date(value).toLocaleString('zh-CN')
}

async function fetchUsers() {
  loading.value = true
  try {
    const data = await listUsers()
    users.value = data.items
  } catch {
    ElMessage.error('加载用户列表失败')
  } finally {
    loading.value = false
  }
}

function resetCreateForm() {
  createForm.username = ''
  createForm.password = ''
  createForm.is_admin = false
  createFormRef.value?.clearValidate()
}

function resetPasswordForm() {
  resetForm.password = ''
  resetTarget.value = null
  resetFormRef.value?.clearValidate()
}

async function handleCreate() {
  const valid = await createFormRef.value?.validate().catch(() => false)
  if (!valid) return

  submitting.value = true
  try {
    await createUser({
      username: createForm.username,
      password: createForm.password,
      is_admin: createForm.is_admin,
    })
    ElMessage.success('用户已创建')
    showCreateDialog.value = false
    await fetchUsers()
  } catch (err: unknown) {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    ElMessage.error(detail || '创建失败')
  } finally {
    submitting.value = false
  }
}

async function toggleActive(user: AuthUser) {
  try {
    await updateUser(user.id, { is_active: !user.is_active })
    ElMessage.success(user.is_active ? '已禁用' : '已启用')
    await fetchUsers()
  } catch {
    ElMessage.error('操作失败')
  }
}

function openResetPassword(user: AuthUser) {
  resetTarget.value = user
  showResetDialog.value = true
}

async function handleResetPassword() {
  const valid = await resetFormRef.value?.validate().catch(() => false)
  if (!valid || !resetTarget.value) return

  submitting.value = true
  try {
    await updateUser(resetTarget.value.id, { password: resetForm.password })
    ElMessage.success('密码已重置')
    showResetDialog.value = false
  } catch {
    ElMessage.error('重置失败')
  } finally {
    submitting.value = false
  }
}

async function handleDelete(id: string) {
  try {
    await deleteUser(id)
    ElMessage.success('用户已删除')
    await fetchUsers()
  } catch (err: unknown) {
    const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    ElMessage.error(detail || '删除失败')
  }
}

onMounted(fetchUsers)
</script>
