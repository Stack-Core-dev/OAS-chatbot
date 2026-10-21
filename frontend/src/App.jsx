import { useEffect, useId, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { streamMessage } from './api'
import './App.css'

function TypingDots() {
  return (
    <span className="typing" aria-label="Assistant is typing">
      <span className="dot" />
      <span className="dot" />
      <span className="dot" />
    </span>
  )
}

function Robot() {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const bodyId = `robot-body-${uid}`
  const headId = `robot-head-${uid}`

  return (
    <svg viewBox="0 0 240 300" aria-hidden="true" focusable="false">
      <defs>
        <linearGradient id={bodyId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#12a3c4" />
          <stop offset="1" stopColor="#0b5d75" />
        </linearGradient>
        <linearGradient id={headId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="1" stopColor="#d9ecf4" />
        </linearGradient>
      </defs>
      <ellipse className="shadow" cx="120" cy="286" rx="52" ry="8" fill="rgba(8,50,66,0.25)" />
      <g className="hov">
        <line
          x1="120"
          y1="58"
          x2="120"
          y2="36"
          stroke="#0b5d75"
          strokeWidth="5"
          strokeLinecap="round"
        />
        <circle className="tip" cx="120" cy="30" r="8" fill="#2fe39a" />
        <g>
          <rect x="46" y="168" width="22" height="58" rx="11" fill={`url(#${bodyId})`} />
          <circle cx="57" cy="230" r="13" fill="#e9f6fb" stroke="#0b5d75" strokeWidth="3" />
        </g>
        <rect x="70" y="150" width="100" height="100" rx="34" fill={`url(#${bodyId})`} />
        <circle cx="120" cy="198" r="16" fill="#e9f6fb" />
        <circle className="core" cx="120" cy="198" r="8" fill="#12a3c4" />
        <rect x="44" y="94" width="12" height="32" rx="6" fill="#0b5d75" />
        <rect x="184" y="94" width="12" height="32" rx="6" fill="#0b5d75" />
        <rect
          x="54"
          y="56"
          width="132"
          height="104"
          rx="40"
          fill={`url(#${headId})`}
          stroke="#0b5d75"
          strokeWidth="4"
        />
        <rect x="70" y="74" width="100" height="66" rx="26" fill="#0b3b4c" />
        <g className="eyes">
          <ellipse cx="101" cy="103" rx="9" ry="12" fill="#5fe3ff" />
          <ellipse cx="139" cy="103" rx="9" ry="12" fill="#5fe3ff" />
        </g>
        <path
          d="M104 124 Q120 136 136 124"
          fill="none"
          stroke="#5fe3ff"
          strokeWidth="4"
          strokeLinecap="round"
        />
        <g className="arm">
          <rect x="171" y="168" width="22" height="58" rx="11" fill={`url(#${bodyId})`} />
          <circle cx="182" cy="230" r="13" fill="#e9f6fb" stroke="#0b5d75" strokeWidth="3" />
        </g>
      </g>
    </svg>
  )
}

function Welcome() {
  const [wave, setWave] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setWave(true), 700)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className={`welcome welcome-go ${wave ? 'welcome-wave' : ''}`}>
      <div className="robot hero-robot">
        <Robot />
      </div>
    </div>
  )
}

function MessageContent({ message }) {
  if (message.role !== 'assistant') {
    return message.content
  }

  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: (props) => <a {...props} target="_blank" rel="noreferrer" />,
        }}
      >
        {message.content}
      </ReactMarkdown>
    </div>
  )
}

function createSessionId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID()
  }
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

function App() {
  const [isOpen, setIsOpen] = useState(false)
  const [isClosing, setIsClosing] = useState(false)
  const [isMaximized, setIsMaximized] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [sessionId, setSessionId] = useState(createSessionId)
  const listRef = useRef(null)
  const reasoningRef = useRef(null)
  const inputRef = useRef(null)
  const closeTimerRef = useRef(null)

  useEffect(() => {
    if (reasoningRef.current) {
      reasoningRef.current.scrollTop = reasoningRef.current.scrollHeight
    }
  }, [messages])

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight
    }
  }, [messages, isOpen, isMaximized])

  useEffect(() => {
    if (!isOpen) return undefined
    const timer = setTimeout(() => inputRef.current?.focus(), 380)
    return () => clearTimeout(timer)
  }, [isOpen])

  useEffect(() => () => clearTimeout(closeTimerRef.current), [])

  function openChat() {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current)
      closeTimerRef.current = null
    }
    setIsClosing(false)
    setIsOpen(true)
  }

  function closeChat() {
    setIsClosing(true)
    closeTimerRef.current = setTimeout(() => {
      setIsOpen(false)
      setIsClosing(false)
      closeTimerRef.current = null
    }, 320)
  }

  function startNewChat() {
    if (sending) return
    setMessages([])
    setInput('')
    setSessionId(createSessionId())
  }

  function handleLauncherClick() {
    if (isOpen) closeChat()
    else openChat()
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const text = input.trim()
    if (!text || sending) return

    const history = messages
      .filter((message) => !message.streaming)
      .map(({ role, content }) => ({ role, content }))

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text },
      {
        role: 'assistant',
        content: '',
        reasoning: '',
        thinking: true,
        streaming: true,
        reasoningOpen: false,
      },
    ])
    setInput('')
    setSending(true)

    try {
      await streamMessage(text, history, {
        onReasoning: (delta) => {
          setMessages((prev) => {
            const last = prev[prev.length - 1]
            if (!last?.streaming) return prev
            return [...prev.slice(0, -1), { ...last, reasoning: (last.reasoning ?? '') + delta }]
          })
        },
        onDelta: (delta) => {
          setMessages((prev) => {
            const last = prev[prev.length - 1]
            if (!last?.streaming) return prev
            return [...prev.slice(0, -1), { ...last, content: last.content + delta, thinking: false }]
          })
        },
      }, sessionId)
    } catch (error) {
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (!last?.streaming) return prev
        return [
          ...prev.slice(0, -1),
          { ...last, content: last.content || `Error: ${error.message}` },
        ]
      })
    } finally {
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (!last?.streaming) return prev
        return [...prev.slice(0, -1), { ...last, streaming: false, thinking: false }]
      })
      setSending(false)
    }
  }

  function toggleReasoning(index) {
    setMessages((prev) =>
      prev.map((message, i) =>
        i === index ? { ...message, reasoningOpen: !message.reasoningOpen } : message,
      ),
    )
  }

  return (
    <div className="widget">
      {isOpen && (
        <section
          className={`chat-box ${isMaximized ? 'maximized' : ''} ${
            isClosing ? 'closing' : ''
          }`}
          aria-label="Chat"
        >
          <header className="chat-header">
            <h2>AWT-Assistant</h2>
            <div className="chat-header-actions">
              <button
                type="button"
                className="new-chat-button"
                onClick={startNewChat}
                disabled={sending}
                aria-label="Start new chat"
                title="Start a new session"
              >
                New chat
              </button>
              <button
                type="button"
                className="icon-button icon-button-maximize"
                onClick={() => setIsMaximized(true)}
                aria-label="Maximize chat"
                title="Maximize"
              >
                &#9633;
              </button>
              <button
                type="button"
                className="icon-button icon-button-minimize"
                onClick={() => setIsMaximized(false)}
                aria-label="Restore chat"
                title="Restore"
              >
                &#10064;
              </button>
              <button
                type="button"
                className="icon-button"
                onClick={closeChat}
                aria-label="Close chat"
              >
                &times;
              </button>
            </div>
          </header>

          <div className="messages" ref={listRef}>
            {messages.length === 0 && <Welcome />}
            {messages.map((message, index) => (
              <div key={index} className={`bubble bubble-${message.role}`}>
                {message.thinking && (
                  <div className="reasoning-live">
                    <div className="reasoning-label">
                      Thinking
                      {!message.reasoning && <TypingDots />}
                    </div>
                    {message.reasoning && (
                      <div className="reasoning-text" ref={reasoningRef}>
                        {message.reasoning}
                      </div>
                    )}
                  </div>
                )}

                {!message.thinking && message.reasoning && (
                  <div className={`reasoning ${message.reasoningOpen ? 'reasoning-open' : ''}`}>
                    <button
                      type="button"
                      className="reasoning-toggle"
                      onClick={() => toggleReasoning(index)}
                      aria-expanded={Boolean(message.reasoningOpen)}
                    >
                      <span className="chevron" aria-hidden="true">
                        &#8250;
                      </span>
                      Reasoning
                    </button>
                    <div className="reasoning-panel">{message.reasoning}</div>
                  </div>
                )}

                {message.streaming && !message.thinking && !message.reasoning && message.content === '' && <TypingDots />}

                <MessageContent message={message} />

                {message.streaming && message.content !== '' && <span className="caret" />}
              </div>
            ))}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Type a message..."
            />
            <button type="submit" disabled={sending || !input.trim()}>
              Send
            </button>
          </form>
        </section>
      )}

      <button
        type="button"
        className={`launcher ${isOpen && !isClosing ? 'launcher-open' : ''}`}
        onClick={handleLauncherClick}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
        aria-expanded={isOpen}
      >
        <svg
          className="i-bot"
          width="32"
          height="32"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#fff"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <rect x="4" y="7" width="16" height="12" rx="4" />
          <path d="M12 7V4" />
          <circle cx="12" cy="3.4" r="1" />
          <circle cx="9" cy="13" r="1.2" fill="#fff" />
          <circle cx="15" cy="13" r="1.2" fill="#fff" />
          <path d="M2 12v3M22 12v3" />
        </svg>
        <svg
          className="i-close"
          width="26"
          height="26"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#fff"
          strokeWidth="2.2"
          strokeLinecap="round"
          aria-hidden="true"
        >
          <path d="M6 6l12 12M18 6 6 18" />
        </svg>
        <i className="launcher-badge" aria-hidden="true" />
      </button>
    </div>
  )
}

export default App
