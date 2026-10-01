# MedRAG+ Frontend

React + Vite + Tailwind CSS v4 client for the MedRAG+ healthcare triage assistant.
It talks to the Flask backend described in `../docs/API.md`.

## Install

```bash
npm install
```

## Develop

Start the backend on `http://localhost:5000`, then:

```bash
npm run dev
```

Open http://localhost:5173. Requests to `/api/*` are proxied to the backend
(see `vite.config.js`).

## Build

```bash
npm run build     # outputs to dist/
npm run preview   # serve the production build locally
```

The production build calls `/api` on the same origin, so serve `dist/` behind the
same host as the backend (or a reverse proxy that forwards `/api`).

## Notes

- The JWT and user profile are stored in `localStorage` (`medrag.token`, `medrag.user`).
  A `401` from any authenticated endpoint logs the user out.
- Voice input uses `MediaRecorder` and needs a secure context (`localhost` or HTTPS)
  plus microphone permission.
- Read-aloud uses the response's `audio` if present, then `POST /api/tts`, then the
  browser's `speechSynthesis` (en-IN / hi-IN / te-IN) as a fallback.
