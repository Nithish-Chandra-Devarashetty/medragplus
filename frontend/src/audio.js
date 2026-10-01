import { api } from './api.js'
import { SPEECH_LANG } from './i18n.js'

let current = null // { audio, url }

export function stopAudio() {
  if (current) {
    try {
      current.audio.pause()
    } catch {
      /* ignore */
    }
    if (current.url) URL.revokeObjectURL(current.url)
    current = null
  }
  try {
    window.speechSynthesis?.cancel()
  } catch {
    /* ignore */
  }
}

function playSrc(src, objectUrl) {
  stopAudio()
  const audio = new Audio(src)
  current = { audio, url: objectUrl }
  return audio.play()
}

export function playBase64(audioObj) {
  const mime = audioObj.mime || 'audio/mpeg'
  return playSrc(`data:${mime};base64,${audioObj.base64}`)
}

function speakBrowser(text, language) {
  return new Promise((resolve, reject) => {
    if (!('speechSynthesis' in window)) {
      reject(new Error('Speech synthesis is not supported in this browser.'))
      return
    }
    stopAudio()
    const u = new SpeechSynthesisUtterance(text)
    u.lang = SPEECH_LANG[language] || 'en-IN'
    const voice = window.speechSynthesis.getVoices().find((v) => v.lang === u.lang || v.lang.startsWith(u.lang.slice(0, 2)))
    if (voice) u.voice = voice
    u.onend = resolve
    u.onerror = resolve
    window.speechSynthesis.speak(u)
  })
}

/** Speak an assistant response: embedded audio → /api/tts → browser speech synthesis. */
export async function speakResponse(data) {
  if (data.audio && data.audio.base64) {
    try {
      await playBase64(data.audio)
      return
    } catch {
      /* fall through */
    }
  }
  const language = data.language || 'en'
  try {
    const res = await api.tts(data.answer, language)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    await playSrc(url, url)
    return
  } catch (e) {
    if (e && e.status === 401) throw e
    /* 503 or playback failure → browser fallback */
  }
  await speakBrowser(data.answer, language)
}
