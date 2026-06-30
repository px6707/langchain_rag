<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <AppHeader />

    <main class="flex-1 max-w-4xl w-full mx-auto px-4 py-6 flex flex-col">
      <div ref="messagesRef" class="flex-1 overflow-y-auto space-y-4 mb-4">
        <div v-if="messages.length === 0" class="text-center text-gray-400 mt-20">
          <el-icon :size="48" class="mb-4"><ChatDotRound /></el-icon>
          <p>开始提问吧，我会基于已上传的文档回答您的问题</p>
        </div>

        <div
          v-for="msg in messages"
          :key="msg.id"
          class="flex"
          :class="msg.role === 'user' ? 'justify-end' : 'justify-start'"
        >
          <div
            class="max-w-[80%] rounded-2xl px-4 py-3"
            :class="
              msg.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-gray-200 text-gray-800'
            "
            >
            <div
              v-if="msg.role === 'assistant' && msg.todos?.length"
              class="mb-3"
            >
              <p class="text-xs text-gray-500 mb-2">任务进度</p>
              <ul class="space-y-1">
                <li
                  v-for="(todo, todoIdx) in msg.todos"
                  :key="todoIdx"
                  class="flex items-start gap-2 text-sm"
                >
                  <el-tag :type="todoStatusType(todo.status)" size="small">
                    {{ todoStatusLabel(todo.status) }}
                  </el-tag>
                  <span class="text-gray-700">{{ todo.content }}</span>
                </li>
              </ul>
            </div>

            <div
              v-if="msg.role === 'assistant' && msg.tool_calls?.length && showToolCalls"
              class="mb-3"
            >
              <el-collapse>
                <el-collapse-item :title="`工具调用 (${msg.tool_calls.length})`">
                  <div class="space-y-2">
                    <div
                      v-for="tool in msg.tool_calls"
                      :key="tool.id"
                      class="text-xs bg-gray-50 rounded p-2 text-gray-600"
                    >
                      <div class="flex items-center gap-2 mb-1">
                        <span class="font-medium text-purple-600">{{ tool.name }}</span>
                        <el-tag
                          size="small"
                          :type="tool.status === 'running' ? 'warning' : 'success'"
                        >
                          {{ tool.status === 'running' ? '调用中' : '已完成' }}
                        </el-tag>
                      </div>
                      <p v-if="tool.args" class="mb-1">
                        <span class="text-gray-500">参数:</span>
                        <span class="font-mono break-all">{{ tool.args }}</span>
                      </p>
                      <p v-if="tool.output" class="whitespace-pre-wrap">
                        <span class="text-gray-500">返回:</span>
                        {{ truncateOutput(tool.output) }}
                      </p>
                      <p v-else-if="tool.status === 'running'" class="text-gray-400">等待返回...</p>
                    </div>
                  </div>
                </el-collapse-item>
              </el-collapse>
            </div>

            <div
              v-if="msg.role === 'assistant' && msg.grounding && msg.grounding.status !== 'skipped'"
              class="mb-2"
            >
              <el-popover placement="top" :width="360" trigger="hover">
                <template #reference>
                  <el-tag :type="groundingTagType(msg.grounding.status)" size="small">
                    {{ groundingLabel(msg.grounding.status) }}
                  </el-tag>
                </template>
                <div class="space-y-2 text-sm">
                  <p class="text-gray-600">
                    支持率: {{ (msg.grounding.supported_ratio * 100).toFixed(0) }}%
                  </p>
                  <ul class="space-y-2">
                    <li
                      v-for="(claim, claimIdx) in msg.grounding.claims"
                      :key="claimIdx"
                      class="border-b border-gray-100 pb-2 last:border-0"
                    >
                      <div class="flex items-center gap-2 mb-1">
                        <el-tag
                          :type="claim.supported ? 'success' : 'danger'"
                          size="small"
                        >
                          {{ claim.supported ? '已支撑' : '未支撑' }}
                        </el-tag>
                        <span v-if="claim.evidence_ref_ids.length" class="text-xs text-gray-400">
                          {{ claim.evidence_ref_ids.join(', ') }}
                        </span>
                      </div>
                      <p class="text-gray-800">{{ claim.claim }}</p>
                      <p v-if="claim.reason" class="text-xs text-gray-500 mt-1">{{ claim.reason }}</p>
                    </li>
                  </ul>
                </div>
              </el-popover>
            </div>

            <CitationText
              v-if="msg.content && msg.role === 'assistant'"
              :content="msg.content"
              @cite-click="openChunkDrawer"
            />
            <p v-else-if="msg.content" class="whitespace-pre-wrap">{{ msg.content }}</p>

            <div
              v-if="msg.role === 'assistant' && msg.sources?.length"
              class="mt-3 pt-3 border-t border-gray-100"
            >
              <p class="text-xs text-gray-500 mb-2">引用来源:</p>
              <div class="space-y-1">
                <div
                  v-for="(source, idx) in msg.sources"
                  :key="idx"
                  class="text-xs bg-gray-50 rounded p-2 text-gray-600"
                >
                  <button
                    type="button"
                    class="font-medium text-blue-600 hover:text-blue-800 underline font-mono"
                    @click="openChunkDrawer(source.document_id, source.chunk_index, source.ref_id)"
                  >
                    [{{ source.ref_id }}]
                  </button>
                  <span class="ml-2">{{ source.filename }}</span>
                  <span
                    v-if="formatSourceTime(source)"
                    class="ml-2 text-gray-500 font-mono"
                  >
                    ({{ formatSourceTime(source) }})
                  </span>
                  <span v-if="source.score != null" class="ml-2 text-gray-400">
                    ({{ source.score.toFixed(3) }})
                  </span>
                  <p class="mt-1 line-clamp-2">{{ source.content }}</p>
                </div>
              </div>
            </div>

            <div
              v-if="msg.role === 'assistant' && msg.run_id && !msg.feedback_submitted"
              class="mt-2 flex justify-end gap-1"
            >
              <el-button
                size="small"
                text
                type="success"
                :loading="feedbackSubmittingId === msg.id"
                @click="submitPositiveFeedback(msg)"
              >
                👍 有帮助
              </el-button>
              <el-button
                size="small"
                text
                type="danger"
                @click="openNegativeFeedbackDialog(msg)"
              >
                👎 不准确
              </el-button>
            </div>
          </div>
        </div>

        <div v-if="loading && !streamingMessageId" class="flex justify-start">
          <div class="bg-white border border-gray-200 rounded-2xl px-4 py-3">
            <el-icon class="is-loading"><Loading /></el-icon>
            <span class="ml-2 text-gray-500">思考中...</span>
          </div>
        </div>
      </div>

      <div class="mb-3 flex items-center justify-end">
        <span class="text-sm text-gray-500 mr-2">显示工具调用</span>
        <el-switch v-model="showToolCalls" @change="handleShowToolCallsChange" />
      </div>

      <div class="bg-white border border-gray-200 rounded-xl p-3 flex gap-3 shadow-sm">
        <el-input
          v-model="input"
          type="textarea"
          :rows="2"
          placeholder="输入您的问题..."
          :disabled="loading || hitlDialogVisible"
          @keydown.enter.exact.prevent="sendMessage"
        />
        <el-button
          type="primary"
          :loading="loading"
          :disabled="!input.trim() || hitlDialogVisible"
          @click="sendMessage"
        >
          发送
        </el-button>
      </div>
    </main>

    <el-dialog
      v-model="hitlDialogVisible"
      :title="isSendEmailHitl ? '确认发送邮件' : '工具调用审批'"
      width="560px"
      :close-on-click-modal="false"
      :close-on-press-escape="false"
      :show-close="false"
    >
      <div v-if="pendingHitlRequest && isSendEmailHitl" class="space-y-3">
        <p class="text-sm text-gray-600">请确认或修改邮件内容与发件账号，确认后将发送。</p>
        <el-form label-position="top">
          <el-form-item label="收件人">
            <el-input v-model="emailForm.to_email" placeholder="recipient@example.com" />
          </el-form-item>
          <el-form-item label="主题">
            <el-input v-model="emailForm.subject" placeholder="邮件主题" />
          </el-form-item>
          <el-form-item label="正文">
            <el-input v-model="emailForm.body" type="textarea" :rows="6" placeholder="邮件正文" />
          </el-form-item>
          <el-form-item label="发件邮箱">
            <el-input v-model="emailForm.smtp_user" placeholder="your@example.com" />
          </el-form-item>
          <el-form-item label="邮箱密码 / 授权码">
            <el-input
              v-model="emailForm.smtp_password"
              type="password"
              show-password
              placeholder="SMTP 密码或授权码"
            />
          </el-form-item>
        </el-form>
      </div>
      <div v-else-if="pendingHitlRequest" class="space-y-4">
        <div
          v-for="(action, index) in pendingHitlRequest.action_requests"
          :key="`${action.name}-${index}`"
          class="rounded-lg border border-gray-200 p-3 bg-gray-50"
        >
          <p class="font-medium text-gray-800 mb-2">{{ action.name }}</p>
          <p v-if="action.description" class="text-sm text-gray-600 whitespace-pre-wrap mb-2">
            {{ action.description }}
          </p>
          <p class="text-xs text-gray-500 mb-1">参数:</p>
          <pre class="text-xs bg-white rounded p-2 overflow-x-auto">{{ formatArgs(action.args) }}</pre>
        </div>
      </div>
      <template #footer>
        <template v-if="isSendEmailHitl">
          <el-button :disabled="hitlSubmitting" @click="handleHitlReject">取消</el-button>
          <el-button
            type="primary"
            :loading="hitlSubmitting"
            :disabled="!isEmailFormValid"
            @click="handleSendEmailConfirm"
          >
            确认发送
          </el-button>
        </template>
        <template v-else>
          <el-button :disabled="hitlSubmitting" @click="handleHitlReject">拒绝</el-button>
          <el-button
            type="primary"
            :loading="hitlSubmitting"
            @click="handleHitlApprove"
          >
            批准
          </el-button>
        </template>
      </template>
    </el-dialog>

    <el-drawer v-model="chunkDrawerVisible" :title="chunkDrawerTitle" size="40%">
      <div v-loading="chunkDrawerLoading" class="min-h-[120px]">
        <div v-if="chunkDrawerIsVideo" class="mb-4">
          <video
            ref="videoRef"
            class="w-full rounded-lg bg-black"
            controls
            :src="chunkDrawerVideoUrl"
            @loadedmetadata="onVideoLoaded"
          />
          <div v-if="chunkDrawerTimePoints.length" class="mt-4">
            <p class="text-xs text-gray-500 mb-2">本回答中的视频时间点</p>
            <ul class="space-y-2 max-h-64 overflow-y-auto">
              <li
                v-for="point in chunkDrawerTimePoints"
                :key="point.refId"
                class="text-xs rounded p-2 cursor-pointer border transition-colors"
                :class="
                  point.refId === chunkDrawerRefId
                    ? 'bg-blue-50 border-blue-300'
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                "
                @click="seekVideo(point.seekSec, point.refId)"
              >
                <div class="flex items-center gap-2 mb-1">
                  <span class="font-mono font-medium text-blue-600">{{ formatTime(point.seekSec) }}</span>
                  <span class="font-mono text-gray-400">[{{ point.refId }}]</span>
                </div>
                <p class="text-gray-600 line-clamp-2">{{ point.preview }}</p>
              </li>
            </ul>
          </div>
        </div>
        <p v-if="chunkDrawerFilename" class="text-sm text-gray-500 mb-3">{{ chunkDrawerFilename }}</p>
        <p v-if="chunkDrawerRefId" class="text-xs font-mono text-blue-600 mb-3">[{{ chunkDrawerRefId }}]</p>
        <p class="whitespace-pre-wrap text-gray-800">{{ chunkDrawerContent }}</p>
      </div>
    </el-drawer>

    <el-dialog
      v-model="feedbackDialogVisible"
      title="反馈：回答不准确"
      width="420px"
      :close-on-click-modal="false"
      @closed="resetFeedbackDialog"
    >
      <el-form label-position="top">
        <el-form-item label="原因" required>
          <el-select v-model="feedbackReason" placeholder="请选择原因" class="w-full">
            <el-option
              v-for="opt in FEEDBACK_REASON_OPTIONS"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="补充说明（可选）">
          <el-input v-model="feedbackComment" type="textarea" :rows="3" maxlength="500" show-word-limit />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="feedbackDialogVisible = false">取消</el-button>
        <el-button
          type="danger"
          :loading="feedbackSubmittingId !== null"
          :disabled="!feedbackReason"
          @click="confirmNegativeFeedback"
        >
          提交反馈
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { ChatDotRound, Loading } from '@element-plus/icons-vue'
import AppHeader from '@/components/AppHeader.vue'
import CitationText from '@/components/CitationText.vue'
import {
  getChatHistory,
  getDocumentChunk,
  getDocumentFileUrl,
  getPendingInterrupt,
  resumeChatStream,
  sendChatStream,
  submitChatFeedback,
  FEEDBACK_REASON_OPTIONS,
  type ChatMessage,
  type FeedbackReason,
  type ChatStreamHandlers,
  type GroundingResult,
  type HITLDecision,
  type HITLRequest,
  type SendEmailArgs,
  type SourceInfo,
  type TodoItem,
  type ToolCallInfo,
} from '@/api'
import { getShowToolCalls, setShowToolCalls } from '@/utils/chatPrefs'
import { getSessionId } from '@/utils/session'

const sessionId = getSessionId()
const messages = ref<ChatMessage[]>([])
const input = ref('')
const loading = ref(false)
const streamingMessageId = ref<string | null>(null)
const showToolCalls = ref(getShowToolCalls())
const messagesRef = ref<HTMLElement>()
const hitlDialogVisible = ref(false)
const pendingHitlRequest = ref<HITLRequest | null>(null)
const hitlSubmitting = ref(false)
const feedbackSubmittingId = ref<string | null>(null)
const feedbackDialogVisible = ref(false)
const feedbackTargetMsg = ref<ChatMessage | null>(null)
const feedbackReason = ref<FeedbackReason | null>(null)
const feedbackComment = ref('')
const emailForm = ref<SendEmailArgs>({
  to_email: '',
  subject: '',
  body: '',
  smtp_user: '',
  smtp_password: '',
})
const chunkDrawerVisible = ref(false)
const chunkDrawerLoading = ref(false)
const chunkDrawerTitle = ref('文档片段')
const chunkDrawerFilename = ref('')
const chunkDrawerRefId = ref('')
const chunkDrawerContent = ref('')
const chunkDrawerIsVideo = ref(false)
const chunkDrawerVideoUrl = ref('')
const chunkDrawerTimePoints = ref<Array<{ refId: string; seekSec: number; preview: string }>>([])
const chunkDrawerPendingSeek = ref<number | null>(null)
const videoRef = ref<HTMLVideoElement | null>(null)

const VIDEO_EXTENSIONS = new Set(['.mp4', '.mkv', '.mov', '.webm', '.avi', '.flv', '.wmv', '.m4v'])

type VideoTimePoint = { refId: string; seekSec: number; preview: string }

function sourceSeekSec(source: SourceInfo): number | null {
  if (source.timestamp_sec != null) return source.timestamp_sec
  if (source.start_sec != null) return source.start_sec
  return null
}

function isVideoSource(source: SourceInfo): boolean {
  if (source.file_type === 'video') return true
  const name = source.filename.toLowerCase()
  const dot = name.lastIndexOf('.')
  if (dot < 0) return false
  return VIDEO_EXTENSIONS.has(name.slice(dot))
}

function formatSourceTime(source: SourceInfo): string | null {
  const sec = sourceSeekSec(source)
  return sec != null ? formatTime(sec) : null
}

function formatTime(sec: number): string {
  const total = Math.max(0, Math.floor(sec))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  }
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function buildVideoTimePoints(sources: SourceInfo[], docId: string): VideoTimePoint[] {
  const points: VideoTimePoint[] = []
  const seenTimes: number[] = []

  for (const source of sources) {
    if (source.document_id !== docId) continue
    const seekSec = sourceSeekSec(source)
    if (seekSec == null) continue
    if (seenTimes.some((t) => Math.abs(t - seekSec) < 3)) continue
    seenTimes.push(seekSec)
    points.push({
      refId: source.ref_id,
      seekSec,
      preview: source.content.slice(0, 80),
    })
  }

  return points.sort((a, b) => a.seekSec - b.seekSec)
}

function onVideoLoaded() {
  if (chunkDrawerPendingSeek.value == null || !videoRef.value) return
  videoRef.value.currentTime = chunkDrawerPendingSeek.value
  void videoRef.value.play().catch(() => {})
  chunkDrawerPendingSeek.value = null
}

function seekVideo(seekSec: number, refId: string) {
  chunkDrawerRefId.value = refId
  chunkDrawerPendingSeek.value = seekSec
  if (videoRef.value) {
    videoRef.value.currentTime = seekSec
    void videoRef.value.play().catch(() => {})
    chunkDrawerPendingSeek.value = null
  }
}

const isSendEmailHitl = computed(
  () => pendingHitlRequest.value?.action_requests[0]?.name === 'send_email',
)

const isEmailFormValid = computed(() =>
  Boolean(
    emailForm.value.to_email.trim()
    && emailForm.value.subject.trim()
    && emailForm.value.body.trim()
    && emailForm.value.smtp_user.trim()
    && emailForm.value.smtp_password.trim(),
  ),
)

function groundingLabel(status: GroundingResult['status']): string {
  if (status === 'supported') return '内容已校验'
  if (status === 'partial') return '部分未支撑'
  if (status === 'not_supported') return '可能缺乏依据'
  return '未校验'
}

function groundingTagType(status: GroundingResult['status']): 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'supported') return 'success'
  if (status === 'partial') return 'warning'
  if (status === 'not_supported') return 'danger'
  return 'info'
}

async function openChunkDrawer(docId: string, chunkIndex: number, refId: string) {
  chunkDrawerVisible.value = true
  chunkDrawerLoading.value = true
  chunkDrawerTitle.value = '文档片段'
  chunkDrawerFilename.value = ''
  chunkDrawerRefId.value = refId
  chunkDrawerContent.value = ''
  chunkDrawerIsVideo.value = false
  chunkDrawerVideoUrl.value = ''
  chunkDrawerTimePoints.value = []
  chunkDrawerPendingSeek.value = null

  const msg = messages.value.find((m) =>
    m.sources?.some((s) => s.ref_id === refId),
  )
  const preview = msg?.sources?.find((s) => s.ref_id === refId)
  if (preview) {
    chunkDrawerFilename.value = preview.filename
    chunkDrawerContent.value = preview.content
    if (isVideoSource(preview)) {
      chunkDrawerIsVideo.value = true
      chunkDrawerVideoUrl.value = getDocumentFileUrl(docId)
      chunkDrawerTimePoints.value = buildVideoTimePoints(msg?.sources ?? [], docId)
      const seek = sourceSeekSec(preview)
      if (seek != null) {
        chunkDrawerPendingSeek.value = seek
      }
    }
  }

  try {
    const chunk = await getDocumentChunk(docId, chunkIndex)
    chunkDrawerFilename.value = chunk.filename
    chunkDrawerRefId.value = chunk.ref_id
    chunkDrawerContent.value = chunk.content
    chunkDrawerTitle.value = chunk.filename

    const chunkAsSource: SourceInfo = {
      document_id: chunk.document_id,
      chunk_index: chunk.chunk_index,
      ref_id: chunk.ref_id,
      filename: chunk.filename,
      content: chunk.content,
      file_type: chunk.file_type,
      content_type: chunk.content_type,
      timestamp_sec: chunk.timestamp_sec,
      start_sec: chunk.start_sec,
      end_sec: chunk.end_sec,
    }
    if (isVideoSource(chunkAsSource)) {
      chunkDrawerIsVideo.value = true
      chunkDrawerVideoUrl.value = getDocumentFileUrl(docId)
      if (msg?.sources?.length) {
        chunkDrawerTimePoints.value = buildVideoTimePoints(msg.sources, docId)
      }
      const seek = sourceSeekSec(chunkAsSource)
      if (seek != null) {
        chunkDrawerPendingSeek.value = seek
      }
    }
  } catch {
    if (!chunkDrawerContent.value) {
      ElMessage.error('无法加载文档片段')
      chunkDrawerVisible.value = false
    }
  } finally {
    chunkDrawerLoading.value = false
  }
}

function truncateOutput(output: string, maxLen = 500): string {
  return output.length > maxLen ? `${output.slice(0, maxLen)}...` : output
}

function formatArgs(args: Record<string, unknown>): string {
  return JSON.stringify(args, null, 2)
}

function emptyEmailForm(): SendEmailArgs {
  return {
    to_email: '',
    subject: '',
    body: '',
    smtp_user: '',
    smtp_password: '',
  }
}

function initEmailFormFromHitl(request: HITLRequest) {
  const args = request.action_requests[0]?.args ?? {}
  emailForm.value = {
    to_email: String(args.to_email ?? ''),
    subject: String(args.subject ?? ''),
    body: String(args.body ?? ''),
    smtp_user: String(args.smtp_user ?? ''),
    smtp_password: String(args.smtp_password ?? ''),
  }
}

function openHitlDialog(request: HITLRequest) {
  pendingHitlRequest.value = request
  if (request.action_requests[0]?.name === 'send_email') {
    initEmailFormFromHitl(request)
  } else {
    emailForm.value = emptyEmailForm()
  }
  hitlDialogVisible.value = true
  loading.value = false
}

function todoStatusLabel(status: TodoItem['status']): string {
  if (status === 'pending') return '待处理'
  if (status === 'in_progress') return '进行中'
  return '已完成'
}

function todoStatusType(status: TodoItem['status']): 'info' | 'warning' | 'success' {
  if (status === 'pending') return 'info'
  if (status === 'in_progress') return 'warning'
  return 'success'
}

function handleShowToolCallsChange(value: string | number | boolean) {
  setShowToolCalls(Boolean(value))
}

function upsertToolCall(assistantId: string, tool: ToolCallInfo) {
  const msg = messages.value.find((m) => m.id === assistantId)
  if (!msg) return
  if (!msg.tool_calls) {
    msg.tool_calls = []
  }
  const index = msg.tool_calls.findIndex((item) => item.id === tool.id)
  if (index >= 0) {
    msg.tool_calls[index] = { ...msg.tool_calls[index], ...tool }
  } else {
    msg.tool_calls.push(tool)
  }
}

function updateTodos(assistantId: string, todos: TodoItem[]) {
  const msg = messages.value.find((m) => m.id === assistantId)
  if (msg) {
    msg.todos = todos
    scrollToBottom()
  }
}

function createStreamHandlers(assistantId: string): ChatStreamHandlers {
  return {
    onToken: (token) => {
      const msg = messages.value.find((m) => m.id === assistantId)
      if (msg) {
        msg.content += token
        scrollToBottom()
      }
    },
    onToolStart: (tool) => {
      upsertToolCall(assistantId, tool)
      scrollToBottom()
    },
    onToolEnd: (tool) => {
      upsertToolCall(assistantId, {
        ...tool,
        status: 'completed',
      })
      scrollToBottom()
    },
    onHitlRequest: (request) => {
      openHitlDialog(request)
    },
    onTodosUpdate: (todos) => {
      updateTodos(assistantId, todos)
    },
    onSources: (sources) => {
      const msg = messages.value.find((m) => m.id === assistantId)
      if (msg) {
        msg.sources = sources
      }
    },
    onGrounding: (grounding) => {
      const msg = messages.value.find((m) => m.id === assistantId)
      if (msg) {
        msg.grounding = grounding
      }
    },
    onTrace: (runId, traceId) => {
      const msg = messages.value.find((m) => m.id === assistantId)
      if (msg) {
        msg.run_id = runId
        msg.trace_id = traceId ?? runId
      }
    },
  }
}

async function submitFeedback(
  msg: ChatMessage,
  kind: 'thumbs_up' | 'thumbs_down',
  reason?: FeedbackReason | null,
  comment?: string | null,
) {
  if (!msg.run_id || msg.feedback_submitted) return
  feedbackSubmittingId.value = msg.id
  try {
    await submitChatFeedback({
      run_id: msg.run_id,
      trace_id: msg.trace_id ?? msg.run_id,
      kind,
      reason: reason ?? undefined,
      comment: comment ?? undefined,
      session_id: sessionId,
    })
    msg.feedback_submitted = true
    ElMessage.success('已记录反馈，感谢！')
  } catch (err: unknown) {
    const detail =
      err && typeof err === 'object' && 'response' in err
        ? (err as { response?: { data?: { detail?: string } } }).response?.data?.detail
        : undefined
    ElMessage.error(typeof detail === 'string' ? detail : '提交反馈失败')
  } finally {
    feedbackSubmittingId.value = null
  }
}

async function submitPositiveFeedback(msg: ChatMessage) {
  await submitFeedback(msg, 'thumbs_up')
}

function openNegativeFeedbackDialog(msg: ChatMessage) {
  feedbackTargetMsg.value = msg
  feedbackReason.value = null
  feedbackComment.value = ''
  feedbackDialogVisible.value = true
}

function resetFeedbackDialog() {
  feedbackTargetMsg.value = null
  feedbackReason.value = null
  feedbackComment.value = ''
}

async function confirmNegativeFeedback() {
  const msg = feedbackTargetMsg.value
  if (!msg || !feedbackReason.value) return
  await submitFeedback(msg, 'thumbs_down', feedbackReason.value, feedbackComment.value || null)
  if (msg.feedback_submitted) {
    feedbackDialogVisible.value = false
  }
}

async function runStream(
  streamFn: (handlers: ChatStreamHandlers) => Promise<void>,
  assistantId: string,
  reloadHistory = true,
) {
  await streamFn(createStreamHandlers(assistantId))
  if (reloadHistory && !hitlDialogVisible.value) {
    await loadHistory()
  }
}

async function loadHistory() {
  try {
    const { messages: history, todos } = await getChatHistory(sessionId)
    messages.value = history
    if (todos?.length) {
      const lastAssistant = [...messages.value].reverse().find((m) => m.role === 'assistant')
      if (lastAssistant) {
        lastAssistant.todos = todos
      }
    }
    scrollToBottom()
  } catch {
    // ignore on first visit
  }
}

async function checkPendingInterrupt() {
  try {
    const { request } = await getPendingInterrupt(sessionId)
    if (!request) return

    const lastAssistant = [...messages.value].reverse().find((m) => m.role === 'assistant')
    if (!lastAssistant) return

    pendingHitlRequest.value = request
    streamingMessageId.value = lastAssistant.id
    openHitlDialog(request)
  } catch {
    // ignore
  }
}

async function sendMessage() {
  const text = input.value.trim()
  if (!text || loading.value || hitlDialogVisible.value) return

  input.value = ''
  loading.value = true

  const tempUserMsg: ChatMessage = {
    id: `temp-${Date.now()}`,
    role: 'user',
    content: text,
    created_at: new Date().toISOString(),
  }
  messages.value.push(tempUserMsg)

  const assistantId = `temp-${Date.now()}-assistant`
  const assistantMsg: ChatMessage = {
    id: assistantId,
    role: 'assistant',
    content: '',
    tool_calls: [],
    created_at: new Date().toISOString(),
  }
  messages.value.push(assistantMsg)
  streamingMessageId.value = assistantId
  scrollToBottom()

  try {
    await runStream(
      (handlers) => sendChatStream(sessionId, text, handlers),
      assistantId,
    )
  } catch {
    if (!hitlDialogVisible.value) {
      ElMessage.error('发送失败，请稍后重试')
      messages.value = messages.value.filter((m) => m.id !== assistantId)
      const lastUser = messages.value[messages.value.length - 1]
      if (lastUser?.role === 'user' && lastUser.content === text) {
        messages.value.pop()
      }
    }
  } finally {
    if (!hitlDialogVisible.value) {
      loading.value = false
      streamingMessageId.value = null
    }
  }
}

async function submitHitlDecisions(decisions: HITLDecision[]) {
  if (!pendingHitlRequest.value || !streamingMessageId.value) return

  const assistantId = streamingMessageId.value
  hitlSubmitting.value = true
  loading.value = true

  try {
    await runStream(
      (handlers) => resumeChatStream(sessionId, decisions, handlers),
      assistantId,
    )
    hitlDialogVisible.value = false
    pendingHitlRequest.value = null
    emailForm.value = emptyEmailForm()
  } catch {
    ElMessage.error('审批处理失败，请重试')
  } finally {
    hitlSubmitting.value = false
    loading.value = false
    streamingMessageId.value = null
  }
}

async function handleHitlApprove() {
  const count = pendingHitlRequest.value?.action_requests.length ?? 0
  const decisions: HITLDecision[] = Array.from({ length: count }, () => ({ type: 'approve' }))
  await submitHitlDecisions(decisions)
}

async function handleHitlReject() {
  const count = pendingHitlRequest.value?.action_requests.length ?? 0
  const decisions: HITLDecision[] = Array.from({ length: count }, () => ({ type: 'reject' }))
  await submitHitlDecisions(decisions)
}

async function handleSendEmailConfirm() {
  if (!isEmailFormValid.value || !pendingHitlRequest.value) return

  const count = pendingHitlRequest.value.action_requests.length
  const form = { ...emailForm.value }
  const decisions: HITLDecision[] = Array.from({ length: count }, () => ({
    type: 'edit',
    edited_action: {
      name: 'send_email',
      args: { ...form },
    },
  }))
  await submitHitlDecisions(decisions)
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}

onMounted(async () => {
  await loadHistory()
  await checkPendingInterrupt()
})
</script>
