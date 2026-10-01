"""POST /chat, POST /chat/voice, POST /tts"""
from __future__ import annotations

from flask import Blueprint, Response, g, jsonify, request

from backend.api.auth import require_auth, services
from backend.services.chat_service import ChatRequest, SessionNotFound
from backend.services.speech_service import SpeechUnavailable

bp = Blueprint("chat", __name__)


def _flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _run(chat_request: ChatRequest):
    try:
        return jsonify(services().pipeline.run(chat_request))
    except SessionNotFound:
        return jsonify(error="Session not found"), 404
    except SpeechUnavailable as exc:
        return jsonify(error=str(exc)), 503
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@bp.post("/chat")
@require_auth
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(error="message is required"), 400
    if len(message) > 2000:
        return jsonify(error="message is too long (max 2000 characters)"), 400
    return _run(ChatRequest(user_id=g.user["_id"], message=message, session_id=data.get("session_id") or None,
                            language=data.get("language") or "auto", tts=_flag(data.get("tts", False))))


@bp.post("/chat/voice")
@require_auth
def chat_voice():
    upload = request.files.get("audio")
    if upload is None:
        return jsonify(error="audio file is required"), 400
    audio = upload.read()
    if not audio:
        return jsonify(error="audio file is empty"), 400
    return _run(ChatRequest(user_id=g.user["_id"], audio=audio, session_id=request.form.get("session_id") or None,
                            language=request.form.get("language") or "auto",
                            tts=_flag(request.form.get("tts", False))))


@bp.post("/tts")
@require_auth
def tts():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    language = data.get("language") or "en"
    if not text:
        return jsonify(error="text is required"), 400
    if language not in services().settings.supported_languages:
        return jsonify(error=f"Unsupported language: {language}"), 400
    try:
        audio = services().speech.synthesize(text[:3000], language)
    except SpeechUnavailable as exc:
        return jsonify(error=str(exc)), 503
    except Exception as exc:  # e.g. gTTS without internet
        return jsonify(error=f"TTS failed: {exc}"), 503
    return Response(audio, mimetype="audio/mpeg")
