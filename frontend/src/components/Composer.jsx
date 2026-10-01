import { useEffect, useRef, useState } from 'react'
import { t } from '../i18n.js'

function pickMime() {
  if (typeof MediaRecorder === 'undefined') return null
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']
  return candidates.find((m) => MediaRecorder.isTypeSupported?.(m)) || ''
}

export default function Composer({ onSend, onVoice, busy, uiLang, autoSpeak, onAutoSpeakChange }) {
  const [text, setText] = useState('')
  const [recording, setRecording] = useState(false)
  const [recSeconds, setRecSeconds] = useState(0)
  const [micError, setMicError] = useState('')
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const discardRef = useRef(false)
  const textareaRef = useRef(null)

  // Auto-grow textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }, [text])

  useEffect(() => {
    if (!recording) return
    const start = Date.now()
    setRecSeconds(0)
    const id = setInterval(() => setRecSeconds(Math.floor((Date.now() - start) / 1000)), 250)
    return () => clearInterval(id)
  }, [recording])

  // Release the mic on unmount
  useEffect(
    () => () => {
      discardRef.current = true
      try {
        recorderRef.current?.state === 'recording' && recorderRef.current.stop()
      } catch {
        /* ignore */
      }
      streamRef.current?.getTracks().forEach((tr) => tr.stop())
    },
    [],
  )

  const submit = () => {
    const msg = text.trim()
    if (!msg || busy) return
    onSend(msg)
    setText('')
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      submit()
    }
  }

  const startRecording = async () => {
    setMicError('')
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setMicError('Voice recording is not supported in this browser.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mimeType = pickMime()
      const rec = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      chunksRef.current = []
      discardRef.current = false
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
      }
      rec.onstop = () => {
        stream.getTracks().forEach((tr) => tr.stop())
        streamRef.current = null
        setRecording(false)
        if (discardRef.current) return
        const type = rec.mimeType || mimeType || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type })
        if (blob.size < 500) {
          setMicError('Recording was too short. Hold on a little longer.')
          return
        }
        onVoice(blob)
      }
      recorderRef.current = rec
      rec.start()
      setRecording(true)
    } catch (err) {
      setMicError(err?.name === 'NotAllowedError' ? 'Microphone permission was denied.' : 'Could not access the microphone.')
    }
  }

  const stopRecording = (discard = false) => {
    discardRef.current = discard
    try {
      recorderRef.current?.stop()
    } catch {
      setRecording(false)
    }
  }

  const fmt = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

  return (
    <div className="border-t border-slate-200 bg-white px-3 pt-2 pb-2 sm:px-4">
      <div className="mx-auto max-w-3xl">
        <div className="mb-1.5 flex items-center justify-between gap-2 text-xs text-slate-600">
          <label className="inline-flex cursor-pointer items-center gap-2 select-none">
            <span className="relative inline-flex">
              <input
                type="checkbox"
                className="peer sr-only"
                checked={autoSpeak}
                onChange={(e) => onAutoSpeakChange(e.target.checked)}
              />
              <span className="h-5 w-9 rounded-full bg-slate-300 transition peer-checked:bg-teal-600 peer-focus-visible:ring-2 peer-focus-visible:ring-teal-600/40" />
              <span className="absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow transition peer-checked:translate-x-4" />
            </span>
            🔊 {t(uiLang, 'autoSpeak')}
          </label>
          <span className="hidden text-slate-400 sm:inline">Enter ↵ to send · Shift+Enter for new line</span>
        </div>

        {micError && <div className="mb-1.5 rounded-md bg-red-50 px-2 py-1 text-xs text-red-700">{micError}</div>}

        {recording ? (
          <div className="flex items-center gap-2 rounded-xl border-2 border-red-300 bg-red-50 px-3 py-2">
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-red-600" />
            </span>
            <span className="flex-1 text-sm font-medium text-red-800">
              {t(uiLang, 'recording')} <span className="tabular-nums">{fmt(recSeconds)}</span>
            </span>
            <button
              type="button"
              onClick={() => stopRecording(true)}
              className="rounded-lg px-3 py-1.5 text-sm text-slate-600 hover:bg-red-100"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => stopRecording(false)}
              className="rounded-lg bg-red-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-red-700"
            >
              ⏹ {t(uiLang, 'stop')}
            </button>
          </div>
        ) : (
          <div className="flex items-end gap-2">
            <textarea
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder={t(uiLang, 'placeholder')}
              className="min-h-[44px] flex-1 resize-none rounded-xl border border-slate-300 bg-slate-50 px-3 py-2.5 text-[15px] outline-none focus:border-teal-600 focus:bg-white focus:ring-2 focus:ring-teal-600/20"
              aria-label="Message"
            />
            <button
              type="button"
              onClick={startRecording}
              disabled={busy}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-slate-300 bg-white text-lg transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50"
              aria-label="Record voice message"
              title="Record voice message"
            >
              🎤
            </button>
            <button
              type="button"
              onClick={submit}
              disabled={busy || !text.trim()}
              className="h-11 shrink-0 rounded-xl bg-teal-700 px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {t(uiLang, 'send')}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
