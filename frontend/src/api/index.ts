import axios from 'axios'
import { getAccessToken, logout } from '@/stores/auth'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

api.interceptors.request.use((config) => {
  const token = getAccessToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      logout()
      const redirect = encodeURIComponent(window.location.pathname + window.location.search)
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = `/login?redirect=${redirect}`
      }
    }
    return Promise.reject(error)
  },
)

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = getAccessToken()
  if (token) {
    headers.Authorization = `Bearer ${token}`
  }
  return headers
}

export interface Document {
  id: string
  filename: string
  file_type: string
  chunk_count: number
  status: string
  error_message?: string | null
  created_at: string
}

export interface DocumentListResponse {
  items: Document[]
  total: number
}

export interface DocumentChunk {
  document_id: string
  chunk_index: number
  ref_id: string
  filename: string
  content: string
}

export interface SourceInfo {
  document_id: string
  chunk_index: number
  ref_id: string
  filename: string
  content: string
  score?: number | null
}

export interface ClaimVerdict {
  claim: string
  supported: boolean
  evidence_ref_ids: string[]
  reason: string
}

export interface GroundingResult {
  status: 'supported' | 'partial' | 'not_supported' | 'skipped'
  supported_ratio: number
  claims: ClaimVerdict[]
}

export interface ChatResponse {
  answer: string
  sources: SourceInfo[]
}

export interface ToolCallInfo {
  id: string
  name: string
  args?: string | null
  output?: string | null
  status?: 'running' | 'completed'
}

export interface HITLActionRequest {
  name: string
  args: Record<string, unknown>
  description?: string | null
}

export interface HITLReviewConfig {
  action_name: string
  allowed_decisions: string[]
}

export interface HITLRequest {
  action_requests: HITLActionRequest[]
  review_configs: HITLReviewConfig[]
}

export type HITLDecision =
  | { type: 'approve' }
  | { type: 'reject'; message?: string }
  | { type: 'edit'; edited_action: { name: string; args: Record<string, unknown> } }

export interface SendEmailArgs {
  to_email: string
  subject: string
  body: string
  smtp_user: string
  smtp_password: string
}

export interface TodoItem {
  content: string
  status: 'pending' | 'in_progress' | 'completed'
}

export interface ChatMessage {
  id: string
  role: string
  content: string
  sources?: SourceInfo[] | null
  grounding?: GroundingResult | null
  tool_calls?: ToolCallInfo[] | null
  todos?: TodoItem[] | null
  created_at: string
}

export interface ChatHistoryResponse {
  messages: ChatMessage[]
  todos?: TodoItem[] | null
}

export interface ChatInterruptResponse {
  request: HITLRequest | null
}

export async function uploadDocument(file: File): Promise<Document> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post<Document>('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function listDocuments(skip = 0, limit = 20): Promise<DocumentListResponse> {
  const { data } = await api.get<DocumentListResponse>('/documents', {
    params: { skip, limit },
  })
  return data
}

export async function deleteDocument(id: string): Promise<void> {
  await api.delete(`/documents/${id}`)
}

export async function getDocumentChunk(docId: string, chunkIndex: number): Promise<DocumentChunk> {
  const { data } = await api.get<DocumentChunk>(`/documents/${docId}/chunks/${chunkIndex}`)
  return data
}

export type ChatStreamEvent =
  | { type: 'token'; content: string }
  | { type: 'tool_start'; id: string; name: string; args?: string | null }
  | { type: 'tool_end'; id: string; name: string; output: string }
  | { type: 'hitl_request'; request: HITLRequest }
  | { type: 'todos_update'; todos: TodoItem[] }
  | { type: 'sources'; sources: SourceInfo[] }
  | { type: 'grounding'; grounding: GroundingResult }
  | { type: 'done' }
  | { type: 'error'; message: string }

export interface ChatStreamHandlers {
  onToken: (token: string) => void
  onToolStart?: (tool: ToolCallInfo) => void
  onToolEnd?: (tool: Pick<ToolCallInfo, 'id' | 'name' | 'output'>) => void
  onHitlRequest?: (request: HITLRequest) => void
  onTodosUpdate?: (todos: TodoItem[]) => void
  onSources?: (sources: SourceInfo[]) => void
  onGrounding?: (grounding: GroundingResult) => void
  onDone?: () => void
  onError?: (message: string) => void
}

async function consumeChatStream(response: Response, handlers: ChatStreamHandlers): Promise<void> {
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `HTTP ${response.status}`)
  }

  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error('无法读取流式响应')
  }

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const event = JSON.parse(line.slice(6)) as ChatStreamEvent

      if (event.type === 'token') {
        handlers.onToken(event.content)
      } else if (event.type === 'tool_start') {
        handlers.onToolStart?.({
          id: event.id,
          name: event.name,
          args: event.args,
          status: 'running',
        })
      } else if (event.type === 'tool_end') {
        handlers.onToolEnd?.({
          id: event.id,
          name: event.name,
          output: event.output,
        })
      } else if (event.type === 'hitl_request') {
        handlers.onHitlRequest?.(event.request)
      } else if (event.type === 'todos_update') {
        handlers.onTodosUpdate?.(event.todos)
      } else if (event.type === 'sources') {
        handlers.onSources?.(event.sources)
      } else if (event.type === 'grounding') {
        handlers.onGrounding?.(event.grounding)
      } else if (event.type === 'done') {
        handlers.onDone?.()
      } else if (event.type === 'error') {
        handlers.onError?.(event.message)
        throw new Error(event.message)
      }
    }
  }
}

export async function sendChatStream(
  sessionId: string,
  message: string,
  handlers: ChatStreamHandlers,
): Promise<void> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ session_id: sessionId, message }),
  })
  await consumeChatStream(response, handlers)
}

export async function resumeChatStream(
  sessionId: string,
  decisions: HITLDecision[],
  handlers: ChatStreamHandlers,
): Promise<void> {
  const response = await fetch('/api/chat/resume', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ session_id: sessionId, decisions }),
  })
  await consumeChatStream(response, handlers)
}

export async function getPendingInterrupt(sessionId: string): Promise<ChatInterruptResponse> {
  const { data } = await api.get<ChatInterruptResponse>('/chat/interrupt', {
    params: { session_id: sessionId },
  })
  return data
}

export async function getChatHistory(sessionId: string): Promise<ChatHistoryResponse> {
  const { data } = await api.get<ChatHistoryResponse>('/chat/history', {
    params: { session_id: sessionId },
  })
  return data
}

export default api
