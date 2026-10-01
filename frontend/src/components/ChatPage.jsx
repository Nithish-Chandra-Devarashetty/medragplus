import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { speakResponse, stopAudio } from '../audio.js'
import { t } from '../i18n.js'
import Composer from './Composer.jsx'
import LanguageSelect from './LanguageSelect.jsx'
import { AssistantBubble, UserBubble } from './MessageBubble.jsx'
import Sidebar from './Sidebar.jsx'

let localId = 0
const nextId = () => `m${++localId}`

function turnsToMessages(turns) {
  const out = []
  for (const turn of turns || []) {
    const q = turn.query || {}
    out.push({ id: nextId(), role: 'user', text: q.transcript || q.original || '', voice: !!q.transcript })
    out.push({ id: nextId(), role: 'assistant', data: turn })
  }
  return out
}

function ThinkingIndicator({ since, uiLang }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(id)
  }, [])
  const secs = Math.max(0, Math.floor((now - since) / 1000))
  return (
    <div className="flex justify-start" aria-live="polite">
      <div className="flex items-center gap-3 rounded-2xl rounded-bl-md border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <span className="flex gap-1">
          <span className="dot h-2 w-2 rounded-full bg-teal-600" />
          <span className="dot h-2 w-2 rounded-full bg-teal-600" />
          <span className="dot h-2 w-2 rounded-full bg-teal-600" />
        </span>
        <span className="text-sm text-slate-600">
          {t(uiLang, 'thinking')} <span className="tabular-nums">{secs}s</span>
        </span>
        {secs >= 15 && <span className="hidden text-xs text-slate-400 sm:inline">Running on CPU — this can take a minute.</span>}
      </div>
    </div>
  )
}

export default function ChatPage({ user, onLogout }) {
  const [language, setLanguage] = useState(user.language || 'auto')
  const [autoSpeak, setAutoSpeak] = useState(false)
  const [sessions, setSessions] = useState([])
  const [sessionsLoading, setSessionsLoading] = useState(true)
  const [sessionsError, setSessionsError] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [pendingSince, setPendingSince] = useState(null)
  const [loadingSession, setLoadingSession] = useState(false)
  const [error, setError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const reqRef = useRef(0) // increments whenever the active conversation changes
  const scrollRef = useRef(null)

  const uiLang = language === 'auto' ? user.language || 'en' : language
  const busy = pendingSince !== null

  const refreshSessions = useCallback(async () => {
    setSessionsLoading(true)
    try {
      const data = await api.history()
      setSessions(Array.isArray(data?.sessions) ? data.sessions : [])
      setSessionsError('')
    } catch (e) {
      if (e.status !== 401) setSessionsError(e.message)
    } finally {
      setSessionsLoading(false)
    }
  }, [])

  useEffect(() => {
    refreshSessions()
  }, [refreshSessions])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages, pendingSince])

  useEffect(() => () => stopAudio(), [])

  const newChat = () => {
    reqRef.current++
    stopAudio()
    setSessionId(null)
    setMessages([])
    setPendingSince(null)
    setError('')
    setSidebarOpen(false)
  }

  const selectSession = async (id) => {
    setSidebarOpen(false)
    if (id === sessionId && !loadingSession) return
    const token = ++reqRef.current
    stopAudio()
    setSessionId(id)
    setMessages([])
    setPendingSince(null)
    setError('')
    setLoadingSession(true)
    try {
      const data = await api.session(id)
      if (token !== reqRef.current) return
      setMessages(turnsToMessages(data?.turns))
    } catch (e) {
      if (token === reqRef.current && e.status !== 401) setError(e.message)
    } finally {
      if (token === reqRef.current) setLoadingSession(false)
    }
  }

  const deleteSession = async (id) => {
    if (!window.confirm('Delete this conversation? This cannot be undone.')) return
    try {
      await api.deleteSession(id)
      setSessions((s) => s.filter((x) => x.session_id !== id))
      if (id === sessionId) newChat()
    } catch (e) {
      if (e.status !== 401) setSessionsError(e.message)
    }
  }

  // Shared handler for text & voice: `send` performs the request, `userMsgId` is the bubble to update.
  const runTurn = async (send, userMsgId) => {
    const token = reqRef.current
    setError('')
    setPendingSince(Date.now())
    try {
      const data = await send()
      if (token !== reqRef.current) {
        refreshSessions()
        return
      }
      if (!data || typeof data.answer !== 'string') throw new Error('Unexpected server response.')
      setMessages((prev) => [
        ...prev.map((m) =>
          m.id === userMsgId && m.voice
            ? { ...m, text: data.query?.transcript || data.query?.original || m.text }
            : m,
        ),
        { id: nextId(), role: 'assistant', data },
      ])
      if (data.session_id) setSessionId(data.session_id)
      refreshSessions()
      if (autoSpeak) speakResponse(data).catch(() => {})
    } catch (e) {
      if (token !== reqRef.current || e.status === 401) return
      setError(e.message || 'Something went wrong.')
    } finally {
      if (token === reqRef.current) setPendingSince(null)
    }
  }

  const sendText = (text) => {
    const id = nextId()
    setMessages((prev) => [...prev, { id, role: 'user', text, voice: false }])
    const payload = { message: text, language, tts: autoSpeak }
    if (sessionId) payload.session_id = sessionId
    runTurn(() => api.chat(payload), id)
  }

  const sendVoice = (blob) => {
    const id = nextId()
    setMessages((prev) => [...prev, { id, role: 'user', text: '', voice: true }])
    const ext = blob.type.includes('ogg') ? 'ogg' : blob.type.includes('mp4') ? 'm4a' : 'webm'
    const form = new FormData()
    form.append('audio', blob, `recording.${ext}`)
    if (sessionId) form.append('session_id', sessionId)
    form.append('language', language)
    form.append('tts', autoSpeak ? 'true' : 'false')
    runTurn(() => api.chatVoice(form), id)
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-2 border-b border-slate-200 bg-white px-3 py-2.5 sm:px-4">
        <button
          type="button"
          onClick={() => setSidebarOpen(true)}
          className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 md:hidden"
          aria-label="Open conversations"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" className="h-5 w-5" aria-hidden="true">
            <path fillRule="evenodd" d="M2 4.75A.75.75 0 0 1 2.75 4h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 4.75Zm0 10.5a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75a.75.75 0 0 1-.75-.75ZM2 10a.75.75 0 0 1 .75-.75h14.5a.75.75 0 0 1 0 1.5H2.75A.75.75 0 0 1 2 10Z" clipRule="evenodd" />
          </svg>
        </button>
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-teal-700 text-lg font-bold text-white">+</div>
        <h1 className="min-w-0 flex-1 leading-tight">
          <span className="block truncate text-base font-bold text-slate-900 sm:inline">MedRAG+</span>
          <span className="block truncate text-xs text-slate-500 sm:inline sm:text-base sm:font-medium sm:text-slate-700">
            <span className="hidden sm:inline"> — </span>Healthcare Triage Assistant
          </span>
        </h1>
        <LanguageSelect value={language} onChange={setLanguage} />
        <div className="hidden items-center gap-2 border-l border-slate-200 pl-3 lg:flex">
          <span className="max-w-[10rem] truncate text-sm text-slate-600" title={user.email}>
            {user.name || user.email}
          </span>
        </div>
        <button
          type="button"
          onClick={onLogout}
          className="shrink-0 rounded-lg border border-slate-300 px-2.5 py-1.5 text-sm text-slate-700 hover:bg-slate-100"
          title={t(uiLang, 'logout')}
        >
          <span className="hidden sm:inline">{t(uiLang, 'logout')}</span>
          <span className="sm:hidden" aria-label={t(uiLang, 'logout')}>⎋</span>
        </button>
      </header>

      <div className="flex min-h-0 flex-1">
        <Sidebar
          sessions={sessions}
          activeId={sessionId}
          loading={sessionsLoading}
          error={sessionsError}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onSelect={selectSession}
          onNew={newChat}
          onDelete={deleteSession}
          uiLang={uiLang}
        />

        <main className="flex min-w-0 flex-1 flex-col">
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-4 sm:px-6">
            <div className="mx-auto flex max-w-3xl flex-col gap-4">
              {messages.length === 0 && !loadingSession && !busy && (
                <div className="mx-auto mt-10 max-w-md text-center sm:mt-20">
                  <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-100 text-3xl">🩺</div>
                  <h2 className="text-xl font-semibold text-slate-900">{t(uiLang, 'welcome')}</h2>
                  <p className="mt-2 text-sm text-slate-600">{t(uiLang, 'welcomeSub')}</p>
                </div>
              )}
              {loadingSession && <p className="text-center text-sm text-slate-500">Loading conversation…</p>}
              {messages.map((m) =>
                m.role === 'user' ? (
                  <UserBubble key={m.id} msg={m} />
                ) : (
                  <AssistantBubble key={m.id} data={m.data} uiLang={uiLang} />
                ),
              )}
              {busy && <ThinkingIndicator since={pendingSince} uiLang={uiLang} />}
              {error && (
                <div role="alert" className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
                  <span aria-hidden="true">⚠️</span>
                  <span className="flex-1 break-words">{error}</span>
                  <button type="button" onClick={() => setError('')} className="text-red-500 hover:text-red-800" aria-label="Dismiss error">
                    ✕
                  </button>
                </div>
              )}
            </div>
          </div>

          <Composer
            onSend={sendText}
            onVoice={sendVoice}
            busy={busy || loadingSession}
            uiLang={uiLang}
            autoSpeak={autoSpeak}
            onAutoSpeakChange={setAutoSpeak}
          />
        </main>
      </div>

      <footer className="border-t border-amber-200 bg-amber-50 px-3 py-1.5 text-center text-[11px] leading-snug text-amber-900 sm:text-xs">
        MedRAG+ provides triage guidance, not a medical diagnosis. In an emergency call 108 / 112.
      </footer>
    </div>
  )
}
