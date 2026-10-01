// Thin fetch wrapper for the MedRAG+ REST API (see docs/API.md).

const TOKEN_KEY = 'medrag.token'
const USER_KEY = 'medrag.user'

export function loadAuth() {
  try {
    const token = localStorage.getItem(TOKEN_KEY)
    const user = JSON.parse(localStorage.getItem(USER_KEY) || 'null')
    return token && user ? { token, user } : null
  } catch {
    return null
  }
}

export function saveAuth(token, user) {
  try {
    localStorage.setItem(TOKEN_KEY, token)
    localStorage.setItem(USER_KEY, JSON.stringify(user))
  } catch {
    /* storage unavailable */
  }
}

export function clearAuth() {
  try {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  } catch {
    /* ignore */
  }
}

function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

let unauthorizedHandler = null
export function onUnauthorized(fn) {
  unauthorizedHandler = fn
}

/**
 * request(path, { method, json, form, raw })
 *  - json: object sent as JSON body
 *  - form: FormData sent as multipart
 *  - raw: return the Response (for binary bodies)
 */
export async function request(path, { method = 'GET', json, form, raw = false, auth = true } = {}) {
  const headers = {}
  let body
  if (json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(json)
  } else if (form) {
    body = form
  }
  const token = auth ? getToken() : null
  if (token) headers.Authorization = `Bearer ${token}`

  let res
  try {
    res = await fetch(`/api${path}`, { method, headers, body })
  } catch {
    throw new ApiError('Cannot reach the server. Is the backend running?', 0)
  }

  if (res.status === 401 && auth && token) {
    if (unauthorizedHandler) unauthorizedHandler()
    throw new ApiError('Your session has expired. Please sign in again.', 401)
  }

  if (!res.ok) {
    let msg = `Request failed (${res.status})`
    try {
      const data = await res.json()
      if (data && data.error) msg = data.error
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(msg, res.status)
  }

  if (raw) return res
  try {
    return await res.json()
  } catch {
    return null
  }
}

export const api = {
  login: (email, password) => request('/login', { method: 'POST', json: { email, password }, auth: false }),
  register: (payload) => request('/register', { method: 'POST', json: payload, auth: false }),
  me: () => request('/me'),
  chat: (payload) => request('/chat', { method: 'POST', json: payload }),
  chatVoice: (form) => request('/chat/voice', { method: 'POST', form }),
  tts: (text, language) => request('/tts', { method: 'POST', json: { text, language }, raw: true }),
  history: () => request('/history'),
  session: (id) => request(`/history/${encodeURIComponent(id)}`),
  deleteSession: (id) => request(`/history/${encodeURIComponent(id)}`, { method: 'DELETE' }),
}
