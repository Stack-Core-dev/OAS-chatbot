const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api'

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`)
  }
  return response.json()
}

export async function streamMessage(message, history = [], handlers = {}) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed with status ${response.status}`)
  }
  if (!response.body) {
    throw new Error('Streaming is not supported in this browser')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() ?? ''

      for (const event of events) {
        for (const line of event.split('\n')) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (!raw) continue

          const data = JSON.parse(raw)
          if (data.error) throw new Error(data.error)
          if (data.reasoning) handlers.onReasoning?.(data.reasoning)
          if (data.delta) handlers.onDelta?.(data.delta)
          if (data.done) return
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
