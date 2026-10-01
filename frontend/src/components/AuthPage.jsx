import { useState } from 'react'
import { api } from '../api.js'
import { LANGUAGES } from '../i18n.js'

export default function AuthPage({ onAuthed, notice }) {
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [language, setLanguage] = useState('en')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const isRegister = mode === 'register'

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    if (isRegister && password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }
    setBusy(true)
    try {
      const data = isRegister
        ? await api.register({ email: email.trim(), password, name: name.trim(), language })
        : await api.login(email.trim(), password)
      if (!data || !data.token || !data.user) throw new Error('Unexpected server response.')
      onAuthed(data)
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setBusy(false)
    }
  }

  const input =
    'w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-[15px] outline-none transition focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20'

  return (
    <div className="flex min-h-full flex-col bg-gradient-to-br from-teal-50 via-slate-50 to-sky-50">
      <main className="flex flex-1 items-center justify-center px-4 py-10">
        <div className="w-full max-w-md">
          <div className="mb-6 text-center">
            <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-700 text-2xl font-bold text-white shadow-md">
              +
            </div>
            <h1 className="text-2xl font-bold text-slate-900">MedRAG+</h1>
            <p className="text-sm text-slate-600">Healthcare Triage Assistant · English · हिन्दी · తెలుగు</p>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <div className="mb-5 grid grid-cols-2 rounded-lg bg-slate-100 p-1 text-sm font-medium">
              {['login', 'register'].map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => {
                    setMode(m)
                    setError('')
                  }}
                  className={`rounded-md py-2 transition ${
                    mode === m ? 'bg-white text-teal-800 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {m === 'login' ? 'Sign in' : 'Create account'}
                </button>
              ))}
            </div>

            {notice && !error && (
              <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
                {notice}
              </div>
            )}

            <form onSubmit={submit} className="space-y-4">
              {isRegister && (
                <label className="block">
                  <span className="mb-1 block text-sm font-medium text-slate-700">Name</span>
                  <input className={input} value={name} onChange={(e) => setName(e.target.value)} required autoComplete="name" />
                </label>
              )}
              <label className="block">
                <span className="mb-1 block text-sm font-medium text-slate-700">Email</span>
                <input
                  className={input}
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-sm font-medium text-slate-700">Password</span>
                <input
                  className={input}
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={isRegister ? 8 : undefined}
                  autoComplete={isRegister ? 'new-password' : 'current-password'}
                />
                {isRegister && <span className="mt-1 block text-xs text-slate-500">At least 8 characters.</span>}
              </label>
              {isRegister && (
                <label className="block">
                  <span className="mb-1 block text-sm font-medium text-slate-700">Preferred language</span>
                  <select className={input} value={language} onChange={(e) => setLanguage(e.target.value)}>
                    {LANGUAGES.filter((l) => l.code !== 'auto').map((l) => (
                      <option key={l.code} value={l.code}>
                        {l.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              {error && (
                <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {error}
                </div>
              )}

              <button
                type="submit"
                disabled={busy}
                className="w-full rounded-lg bg-teal-700 py-2.5 font-semibold text-white shadow-sm transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {busy ? 'Please wait…' : isRegister ? 'Create account' : 'Sign in'}
              </button>
            </form>
          </div>
        </div>
      </main>
      <footer className="border-t border-slate-200 bg-white/80 px-4 py-3 text-center text-xs text-slate-600">
        MedRAG+ provides triage guidance, not a medical diagnosis. In an emergency call 108 / 112.
      </footer>
    </div>
  )
}
