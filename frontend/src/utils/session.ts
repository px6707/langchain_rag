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

export function statusLabel(status: string, parseStage?: string | null): string {
  if (status === 'processing' && parseStage) {
    const stageMap: Record<string, string> = {
      queued: '排队中',
      mineru: 'MinerU 解析',
      python: 'Python 解析',
      python_fallback: 'Python 降级解析',
      asr: '语音转写',
      vlm: '视觉摘要',
      video_extract: '视频抽帧',
      chunk: '分块索引',
    }
    return stageMap[parseStage] || '处理中'
  }
  const map: Record<string, string> = {
    queued: '排队中',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  }
  return map[status] || status
}

export function statusType(status: string): '' | 'success' | 'warning' | 'danger' | 'info' {
  const map: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    queued: 'info',
    processing: 'warning',
    completed: 'success',
    failed: 'danger',
  }
  return map[status] || 'info'
}
