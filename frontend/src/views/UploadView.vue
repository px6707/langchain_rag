<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <AppHeader />

    <main class="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
      <el-card class="mb-6">
        <template #header>
          <span class="font-medium">上传文档</span>
        </template>
        <el-upload
          drag
          :auto-upload="false"
          :show-file-list="false"
          accept=".pdf,.txt,.md,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.png,.jpg,.jpeg,.webp,.gif,.bmp,.tiff,.mp3,.wav,.m4a,.aac,.ogg,.flac,.mp4,.mkv,.mov,.webm"
          :on-change="handleFileChange"
        >
          <el-icon class="el-icon--upload" :size="48"><UploadFilled /></el-icon>
          <div class="el-upload__text">
            拖拽文件到此处，或 <em>点击上传</em>
          </div>
          <template #tip>
            <div class="el-upload__tip text-gray-500">
              支持 PDF、Office、图片、音频、视频等；单文件最大 200MB。上传后立即返回，后台队列解析。
            </div>
          </template>
        </el-upload>
      </el-card>

      <el-card>
        <template #header>
          <div class="flex items-center justify-between">
            <span class="font-medium">文档列表</span>
            <el-button :icon="Refresh" circle size="small" @click="fetchDocuments" />
          </div>
        </template>

        <el-table v-loading="loading" :data="documents" stripe style="width: 100%">
          <el-table-column prop="filename" label="文件名" min-width="200" />
          <el-table-column prop="file_type" label="类型" width="80" />
          <el-table-column prop="chunk_count" label="分块数" width="90" />
          <el-table-column label="状态" width="100">
            <template #default="{ row }">
              <el-tag :type="statusType(row.status)" size="small">
                {{ statusLabel(row.status, row.parse_stage) }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column label="上传时间" width="180">
            <template #default="{ row }">
              {{ formatDate(row.created_at) }}
            </template>
          </el-table-column>
          <el-table-column label="操作" width="80" fixed="right">
            <template #default="{ row }">
              <el-popconfirm title="确定删除此文档？" @confirm="handleDelete(row.id)">
                <template #reference>
                  <el-button type="danger" :icon="Delete" circle size="small" />
                </template>
              </el-popconfirm>
            </template>
          </el-table-column>
        </el-table>

        <div v-if="total > pageSize" class="mt-4 flex justify-end">
          <el-pagination
            v-model:current-page="currentPage"
            :page-size="pageSize"
            :total="total"
            layout="prev, pager, next"
            @current-change="fetchDocuments"
          />
        </div>
      </el-card>
    </main>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { ElMessage, type UploadFile } from 'element-plus'
import { Delete, Refresh, UploadFilled } from '@element-plus/icons-vue'
import AppHeader from '@/components/AppHeader.vue'
import { deleteDocument, listDocuments, uploadDocument, type Document } from '@/api'
import { formatDate, statusLabel, statusType } from '@/utils/session'

const documents = ref<Document[]>([])
const loading = ref(false)
const total = ref(0)
const currentPage = ref(1)
const pageSize = 20
let pollTimer: ReturnType<typeof setInterval> | null = null

async function fetchDocuments() {
  loading.value = true
  try {
    const skip = (currentPage.value - 1) * pageSize
    const data = await listDocuments(skip, pageSize)
    documents.value = data.items
    total.value = data.total
  } catch {
    ElMessage.error('加载文档列表失败')
  } finally {
    loading.value = false
  }
}

async function handleFileChange(uploadFile: UploadFile) {
  if (!uploadFile.raw) return

  try {
    await uploadDocument(uploadFile.raw)
    ElMessage.success('上传成功，已加入解析队列')
    await fetchDocuments()
    startPolling()
  } catch {
    ElMessage.error('上传失败')
  }
}

async function handleDelete(id: string) {
  try {
    await deleteDocument(id)
    ElMessage.success('删除成功')
    await fetchDocuments()
  } catch {
    ElMessage.error('删除失败')
  }
}

function startPolling() {
  if (pollTimer) return
  pollTimer = setInterval(async () => {
    await fetchDocuments()
    const hasProcessing = documents.value.some(
      (d) => d.status === 'queued' || d.status === 'processing',
    )
    if (!hasProcessing && pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }, 3000)
}

onMounted(() => {
  fetchDocuments()
  startPolling()
})

onUnmounted(() => {
  if (pollTimer) {
    clearInterval(pollTimer)
  }
})
</script>
