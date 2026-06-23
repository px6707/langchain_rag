const SHOW_TOOL_CALLS_KEY = 'rag_show_tool_calls'

export function getShowToolCalls(): boolean {
  const value = localStorage.getItem(SHOW_TOOL_CALLS_KEY)
  if (value === null) {
    return true
  }
  return value === 'true'
}

export function setShowToolCalls(value: boolean): void {
  localStorage.setItem(SHOW_TOOL_CALLS_KEY, String(value))
}
