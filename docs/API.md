# MedRAG+ REST API

Base URL: `http://localhost:5000/api` (the Vite dev server proxies `/api` to it).

All endpoints except `/register`, `/login` and `/health` require
`Authorization: Bearer <token>`. Errors are returned as `{"error": "<message>"}`
with a 4xx/5xx status.

Language codes: `en` (English), `hi` (Hindi), `te` (Telugu). Requests may also use
`auto` to let the server detect the language.

Triage levels: `self_care`, `gp_appointment`, `urgent_care`, `emergency`.

Safety actions: `deliver` (answer passed all checks), `rewrite` (unsupported
sentences removed), `escalate` (answer replaced with doctor/emergency advice).

---

## POST /api/register

```json
{ "email": "a@b.com", "password": "min 8 chars", "name": "Asha", "language": "en" }
```

`201` →

```json
{ "token": "<jwt>", "user": { "id": "...", "email": "a@b.com", "name": "Asha", "language": "en" } }
```

`409` if the email already exists, `400` on validation failure.

## POST /api/login

```json
{ "email": "a@b.com", "password": "..." }
```

`200` → same shape as `/register`. `401` on bad credentials.

## GET /api/me

`200` → `{ "user": { ... } }`

## POST /api/chat  (text input)

```json
{
  "session_id": "optional; omit to start a new session",
  "message": "I have a headache since yesterday",
  "language": "auto | en | hi | te",
  "tts": true
}
```

## POST /api/chat/voice  (voice input, multipart/form-data)

Fields: `audio` (file; webm/ogg/wav/mp3), `session_id` (optional),
`language` (optional), `tts` (`"true"`/`"false"`).

### Response (both chat endpoints) `200` →

```json
{
  "session_id": "...",
  "message_id": "...",
  "query": {
    "original": "मुझे कल से सिरदर्द है",
    "transcript": null,
    "detected_language": "hi",
    "english": "I have had a headache since yesterday"
  },
  "language": "hi",
  "answer": "<final answer in the user's language>",
  "answer_english": "<final answer in English>",
  "triage_level": "self_care",
  "action": "deliver",
  "escalated": false,
  "emergency": false,
  "confidence": 0.82,
  "hallucination": false,
  "safety_flags": [],
  "unsupported_claims": [],
  "sources": [
    { "source": "MedlinePlus", "title": "Headache", "url": "https://...", "section": null,
      "page": null, "score": 0.71, "snippet": "first ~240 chars of the chunk" }
  ],
  "audio": { "mime": "audio/mpeg", "base64": "..." },
  "timings_ms": { "asr": 0, "translate_in": 812, "embedding": 40, "retrieval": 12,
                  "generation": 21000, "safety": 300, "translate_out": 900, "tts": 700,
                  "total": 23800 },
  "disclaimer": "MedRAG+ offers triage guidance, not a medical diagnosis.",
  "created_at": "2026-10-01T10:00:00Z"
}
```

`audio` is `null` when `tts` is false or TTS is unavailable; the client may then
fall back to browser speech synthesis. `safety_flags` values include
`emergency`, `low_confidence`, `hallucination`, `no_context`.

## POST /api/tts

```json
{ "text": "...", "language": "hi" }
```

`200` → `audio/mpeg` bytes. `503` if TTS is not configured.

## GET /api/history

`200` →

```json
{ "sessions": [ { "session_id": "...", "title": "first user message (truncated)",
                  "language": "en", "created_at": "...", "updated_at": "...",
                  "message_count": 3 } ] }
```

Newest first.

## GET /api/history/<session_id>

`200` → `{ "session_id": "...", "turns": [ <chat response objects, oldest first, without audio> ] }`

## DELETE /api/history/<session_id>

`200` → `{ "deleted": true }`

## GET /api/health

`200` → `{ "status": "ok", "components": { "llm": "llamacpp", "embeddings": "pubmedbert", ... } }`
