import { useEffect, useRef, useState } from 'react'
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

function App() {
  const [isOpen, setIsOpen] = useState(false)
  const [isMaximized, setIsMaximized] = useState(false)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const listRef = useRef(null)
  const reasoningRef = useRef(null)

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
      })
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
        <section className={`chat-box ${isMaximized ? 'maximized' : ''}`} aria-label="Chat">
          <header className="chat-header">
            <h2>OAS-Assistant</h2>
            <div className="chat-header-actions">
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
                onClick={() => setIsOpen(false)}
                aria-label="Close chat"
              >
                &times;
              </button>
            </div>
          </header>

          <div className="messages" ref={listRef}>
            {messages.length === 0 && (
              <p className="empty">Send a message to start chatting.</p>
            )}
            {messages.map((message, index) => (
              <div key={index} className={`bubble bubble-${message.role}`}>
                {message.thinking && (
                  <div className="reasoning-live">
                    <div className="reasoning-label">
                      Thinking
                      {(!message.reasoning) && <TypingDots />}
                    </div>
                    {message.reasoning && (
                      <div className="reasoning-text" ref={reasoningRef}>
                        {message.reasoning}
                        {message.thinking && <span className="caret" />}
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
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Type a message..."
              autoFocus
            />
            <button type="submit" disabled={sending || !input.trim()}>
              Send
            </button>
          </form>
        </section>
      )}

      <button
        type="button"
        className={`launcher ${isOpen ? 'launcher-open' : ''}`}
        onClick={() => setIsOpen((open) => !open)}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
        aria-expanded={isOpen}
      >
        {isOpen ? (
          <span key="close" className="launcher-icon">
            &times;
          </span>
        ) : (
          <span key="chat" className="launcher-icon">
            &#128172;
          </span>
        )}
      </button>
    </div>
  )
}

export default App
