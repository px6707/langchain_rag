export const SESSION_KEY = 'rag_session_id'

export function getSessionId(): string {
  let sessionId = localStorage.getItem(SESSION_KEY)
  if (!sessionId) {
    sessionId = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, sessionId)
  }
  return sessionId
}

export function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleString('zh-CN')
}

export function statusLabel(status: string): string {
  const map: Record<string, string> = {
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  }
  return map[status] || status
}

export function statusType(status: string): '' | 'success' | 'warning' | 'danger' | 'info' {
  const map: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    processing: 'warning',
    completed: 'success',
    failed: 'danger',
  }
  return map[status] || 'info'
}
