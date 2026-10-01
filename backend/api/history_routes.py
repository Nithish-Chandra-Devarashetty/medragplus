"""GET /history, GET /history/<session_id>, DELETE /history/<session_id>"""
from __future__ import annotations

from flask import Blueprint, g, jsonify

from backend.api.auth import require_auth, services
from backend.db.store import iso

bp = Blueprint("history", __name__)


@bp.get("/history")
@require_auth
def list_sessions():
    sessions = services().store.list_sessions(g.user["_id"])
    return jsonify(sessions=[
        {"session_id": s["_id"], "title": s["title"], "language": s.get("language"),
         "created_at": iso(s["created_at"]), "updated_at": iso(s["updated_at"]),
         "message_count": s.get("message_count", 0)}
        for s in sessions
    ])


@bp.get("/history/<session_id>")
@require_auth
def get_session(session_id: str):
    svc = services()
    if svc.store.get_session(session_id, g.user["_id"]) is None:
        return jsonify(error="Session not found"), 404
    turns = [svc.pipeline.to_response(m) for m in svc.store.get_messages(session_id)]
    return jsonify(session_id=session_id, turns=turns)


@bp.delete("/history/<session_id>")
@require_auth
def delete_session(session_id: str):
    if not services().store.delete_session(session_id, g.user["_id"]):
        return jsonify(error="Session not found"), 404
    return jsonify(deleted=True)
