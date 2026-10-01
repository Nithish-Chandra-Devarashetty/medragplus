import { t } from '../i18n.js'

function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  return sameDay
    ? d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: d.getFullYear() === now.getFullYear() ? undefined : 'numeric' })
}

export default function Sidebar({ sessions, activeId, loading, error, open, onClose, onSelect, onNew, onDelete, uiLang }) {
  return (
    <>
      {/* Mobile backdrop */}
      <div
        className={`fixed inset-0 z-30 bg-slate-900/40 transition-opacity md:hidden ${open ? 'opacity-100' : 'pointer-events-none opacity-0'}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 max-w-[85vw] flex-col border-r border-slate-200 bg-white transition-transform md:static md:z-auto md:max-w-none md:translate-x-0 ${
          open ? 'translate-x-0 shadow-xl' : '-translate-x-full'
        }`}
        aria-label={t(uiLang, 'history')}
      >
        <div className="flex items-center gap-2 border-b border-slate-200 p-3">
          <button
            type="button"
            onClick={onNew}
            className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-teal-700 px-3 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-800"
          >
            <span className="text-lg leading-none">+</span> {t(uiLang, 'newChat')}
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 md:hidden"
            aria-label="Close sidebar"
          >
            ✕
          </button>
        </div>

        <div className="px-3 pt-3 pb-1 text-xs font-semibold tracking-wide text-slate-500 uppercase">{t(uiLang, 'history')}</div>

        <nav className="flex-1 overflow-y-auto px-2 pb-3">
          {loading && sessions.length === 0 && <p className="px-2 py-3 text-sm text-slate-500">Loading…</p>}
          {error && <p className="mx-1 my-2 rounded-md bg-red-50 px-2 py-1.5 text-xs text-red-700">{error}</p>}
          {!loading && !error && sessions.length === 0 && (
            <p className="px-2 py-3 text-sm text-slate-500">{t(uiLang, 'noSessions')}</p>
          )}
          <ul className="space-y-1">
            {sessions.map((s) => {
              const active = s.session_id === activeId
              return (
                <li key={s.session_id} className="group relative">
                  <button
                    type="button"
                    onClick={() => onSelect(s.session_id)}
                    className={`w-full rounded-lg py-2 pr-9 pl-3 text-left transition ${
                      active ? 'bg-teal-50 ring-1 ring-teal-600/30' : 'hover:bg-slate-100'
                    }`}
                  >
                    <div className={`truncate text-sm ${active ? 'font-semibold text-teal-900' : 'text-slate-800'}`}>
                      {s.title || 'Untitled'}
                    </div>
                    <div className="mt-0.5 flex items-center gap-2 text-xs text-slate-500">
                      <span>{formatDate(s.updated_at || s.created_at)}</span>
                      {s.message_count ? <span>· {s.message_count} msg</span> : null}
                      {s.language ? <span className="uppercase">· {s.language}</span> : null}
                    </div>
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(s.session_id)
                    }}
                    className="absolute top-1/2 right-1.5 -translate-y-1/2 rounded-md p-1.5 text-slate-400 transition hover:bg-red-50 hover:text-red-600 md:opacity-0 md:group-hover:opacity-100 md:focus:opacity-100"
                    aria-label="Delete conversation"
                    title="Delete conversation"
                  >
                    <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4" aria-hidden="true">
                      <path
                        fillRule="evenodd"
                        d="M8.75 1A2.75 2.75 0 0 0 6 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 1 0 .23 1.482l.149-.022.841 10.518A2.75 2.75 0 0 0 7.596 19h4.807a2.75 2.75 0 0 0 2.742-2.53l.841-10.52.149.023a.75.75 0 0 0 .23-1.482A41.03 41.03 0 0 0 14 4.193V3.75A2.75 2.75 0 0 0 11.25 1h-2.5ZM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4ZM8.58 7.72a.75.75 0 0 0-1.5.06l.3 7.5a.75.75 0 1 0 1.5-.06l-.3-7.5Zm4.34.06a.75.75 0 1 0-1.5-.06l-.3 7.5a.75.75 0 1 0 1.5.06l.3-7.5Z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </button>
                </li>
              )
            })}
          </ul>
        </nav>
      </aside>
    </>
  )
}
