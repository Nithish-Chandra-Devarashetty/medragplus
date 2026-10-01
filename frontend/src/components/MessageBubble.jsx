import { useState } from 'react'
import { speakResponse, stopAudio } from '../audio.js'
import { t } from '../i18n.js'
import TriageBadge from './TriageBadge.jsx'

function Collapsible({ title, count, children }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-2 border-t border-slate-200/80 pt-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 text-left text-xs font-semibold text-slate-600 hover:text-slate-900"
        aria-expanded={open}
      >
        <span className={`inline-block transition-transform ${open ? 'rotate-90' : ''}`}>▸</span>
        {title}
        {count != null && <span className="rounded-full bg-slate-200 px-1.5 text-[10px] text-slate-700">{count}</span>}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  )
}

function ConfidenceBar({ value, label }) {
  if (typeof value !== 'number' || Number.isNaN(value)) return null
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100)
  const color = pct >= 70 ? 'bg-green-500' : pct >= 40 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2 text-xs text-slate-600" title={`${label}: ${pct}%`}>
      <span>{label}</span>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-200 sm:w-20" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-medium tabular-nums">{pct}%</span>
    </div>
  )
}

function YesNo({ value }) {
  return value ? <span className="font-semibold text-red-700">yes</span> : <span className="text-green-700">no</span>
}

function SpeakButton({ data }) {
  const [state, setState] = useState('idle') // idle | busy | error
  const [err, setErr] = useState('')

  const onClick = async () => {
    if (state === 'busy') {
      stopAudio()
      setState('idle')
      return
    }
    setErr('')
    setState('busy')
    try {
      await speakResponse(data)
      setState('idle')
    } catch (e) {
      setErr(e.message || 'Could not play audio')
      setState('error')
    }
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      <button
        type="button"
        onClick={onClick}
        className={`rounded-full p-1.5 text-base leading-none transition hover:bg-slate-200 ${state === 'busy' ? 'animate-pulse bg-slate-200' : ''}`}
        aria-label={state === 'busy' ? 'Stop audio' : 'Play answer aloud'}
        title={state === 'busy' ? 'Stop' : 'Play answer aloud'}
      >
        {state === 'busy' ? '⏹' : '🔊'}
      </button>
      {err && <span className="text-xs text-red-600">{err}</span>}
    </span>
  )
}

export function UserBubble({ msg }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[88%] rounded-2xl rounded-br-md bg-teal-700 px-4 py-2.5 text-white shadow-sm sm:max-w-[75%]">
        {msg.voice && (
          <div className="mb-1 text-[11px] font-semibold tracking-wide text-teal-100 uppercase">🎤 transcript</div>
        )}
        {msg.text ? (
          <p className="text-[15px] break-words whitespace-pre-wrap">{msg.text}</p>
        ) : (
          <p className="text-[15px] text-teal-100 italic">Voice message…</p>
        )}
      </div>
    </div>
  )
}

export function AssistantBubble({ data, uiLang }) {
  const emergency = data.emergency || data.triage_level === 'emergency'
  const escalated = !!data.escalated
  const sources = Array.isArray(data.sources) ? data.sources : []
  const flags = Array.isArray(data.safety_flags) ? data.safety_flags : []
  const claims = Array.isArray(data.unsupported_claims) ? data.unsupported_claims : []
  const timings = data.timings_ms && typeof data.timings_ms === 'object' ? Object.entries(data.timings_ms) : []

  let frame = 'border border-slate-200 bg-white'
  if (escalated) frame = emergency ? 'border-2 border-red-500 bg-red-50/60' : 'border-2 border-amber-400 bg-amber-50/60'

  return (
    <div className="flex justify-start">
      <div className={`w-full max-w-[94%] overflow-hidden rounded-2xl rounded-bl-md shadow-sm sm:max-w-[80%] ${frame}`}>
        {escalated && (
          <div
            role="alert"
            className={`flex items-center gap-2 px-4 py-2 text-sm font-bold ${
              emergency ? 'bg-red-600 text-white' : 'bg-amber-400 text-amber-950'
            }`}
          >
            <span aria-hidden="true">{emergency ? '🚨' : '⚠️'}</span>
            {emergency ? t(uiLang, 'emergencyBanner') : t(uiLang, 'escalated')}
          </div>
        )}

        <div className="px-4 py-3">
          <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
            <TriageBadge level={data.triage_level} />
            <ConfidenceBar value={data.confidence} label={t(uiLang, 'confidence')} />
            <span className="ml-auto">
              <SpeakButton data={data} />
            </span>
          </div>

          <p lang={data.language || undefined} className="text-[15px] break-words whitespace-pre-wrap text-slate-800">
            {data.answer}
          </p>

          {sources.length > 0 && (
            <Collapsible title={t(uiLang, 'sources')} count={sources.length}>
              <ol className="space-y-2">
                {sources.map((s, i) => (
                  <li key={i} className="rounded-lg bg-slate-50 p-2.5 text-xs ring-1 ring-slate-200">
                    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                      <span className="font-semibold text-slate-500">[{i + 1}]</span>
                      {s.url ? (
                        <a href={s.url} target="_blank" rel="noopener noreferrer" className="font-semibold break-all text-teal-800 underline hover:text-teal-950">
                          {s.title || s.url}
                        </a>
                      ) : (
                        <span className="font-semibold text-slate-800">{s.title || 'Untitled'}</span>
                      )}
                      {s.source && <span className="text-slate-500">· {s.source}</span>}
                      {s.section && <span className="text-slate-500">· {s.section}</span>}
                      {s.page != null && <span className="text-slate-500">· p. {s.page}</span>}
                      {typeof s.score === 'number' && (
                        <span className="ml-auto rounded bg-slate-200 px-1.5 font-mono text-[10px] text-slate-700">score {s.score.toFixed(2)}</span>
                      )}
                    </div>
                    {s.snippet && <p className="mt-1 text-slate-600">{s.snippet}</p>}
                  </li>
                ))}
              </ol>
            </Collapsible>
          )}

          <Collapsible title={t(uiLang, 'safety')}>
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
              <dt className="text-slate-500">Action</dt>
              <dd className="font-mono font-semibold text-slate-800">{data.action || '—'}</dd>
              <dt className="text-slate-500">Flags</dt>
              <dd className="flex flex-wrap gap-1">
                {flags.length === 0 ? (
                  <span className="text-slate-500">none</span>
                ) : (
                  flags.map((f) => (
                    <span key={f} className="rounded bg-slate-200 px-1.5 font-mono text-[11px] text-slate-800">
                      {f}
                    </span>
                  ))
                )}
              </dd>
              <dt className="text-slate-500">Hallucination</dt>
              <dd>
                <YesNo value={data.hallucination} />
              </dd>
              {data.query?.detected_language && (
                <>
                  <dt className="text-slate-500">Detected lang</dt>
                  <dd className="font-mono uppercase">{data.query.detected_language}</dd>
                </>
              )}
              {data.query?.english && data.query.english !== data.query.original && (
                <>
                  <dt className="text-slate-500">Query (EN)</dt>
                  <dd className="text-slate-700">{data.query.english}</dd>
                </>
              )}
            </dl>

            {claims.length > 0 && (
              <div className="mt-2 text-xs">
                <div className="mb-1 text-slate-500">Unsupported claims removed</div>
                <ul className="list-disc space-y-0.5 pl-5 text-slate-700">
                  {claims.map((c, i) => (
                    <li key={i}>{typeof c === 'string' ? c : JSON.stringify(c)}</li>
                  ))}
                </ul>
              </div>
            )}

            {timings.length > 0 && (
              <table className="mt-2 w-full max-w-xs text-xs">
                <thead>
                  <tr className="text-left text-slate-500">
                    <th className="py-0.5 font-medium">Stage</th>
                    <th className="py-0.5 text-right font-medium">ms</th>
                  </tr>
                </thead>
                <tbody>
                  {timings.map(([k, v]) => (
                    <tr key={k} className={`border-t border-slate-200 ${k === 'total' ? 'font-semibold' : ''}`}>
                      <td className="py-0.5 font-mono">{k}</td>
                      <td className="py-0.5 text-right tabular-nums">{typeof v === 'number' ? Math.round(v).toLocaleString() : String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Collapsible>
        </div>
      </div>
    </div>
  )
}
