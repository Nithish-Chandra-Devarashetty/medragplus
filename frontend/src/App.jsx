import { useCallback, useEffect, useState } from 'react'
import { clearAuth, loadAuth, onUnauthorized, saveAuth } from './api.js'
import { stopAudio } from './audio.js'
import AuthPage from './components/AuthPage.jsx'
import ChatPage from './components/ChatPage.jsx'

export default function App() {
  const [auth, setAuth] = useState(() => loadAuth())
  const [notice, setNotice] = useState('')

  const logout = useCallback((message = '') => {
    stopAudio()
    clearAuth()
    setAuth(null)
    setNotice(message)
  }, [])

  useEffect(() => {
    onUnauthorized(() => logout('Your session has expired. Please sign in again.'))
    return () => onUnauthorized(null)
  }, [logout])

  const handleAuthed = ({ token, user }) => {
    saveAuth(token, user)
    setNotice('')
    setAuth({ token, user })
  }

  if (!auth) return <AuthPage onAuthed={handleAuthed} notice={notice} />
  return <ChatPage key={auth.user.id || auth.user.email} user={auth.user} onLogout={() => logout()} />
}
