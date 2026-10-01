"""Password hashing, JWT issuing and the @require_auth decorator."""
from __future__ import annotations

from datetime import timedelta
from functools import wraps

import jwt
from flask import current_app, g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from backend.db.store import utcnow


def services():
    return current_app.extensions["medrag"]


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


def issue_token(user_id: str) -> str:
    settings = services().settings
    now = utcnow()
    payload = {"sub": user_id, "iat": now, "exp": now + timedelta(hours=settings.jwt_expiry_hours)}
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def public_user(user: dict) -> dict:
    return {"id": user["_id"], "email": user["email"], "name": user.get("name"), "language": user.get("language", "en")}


def require_auth(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify(error="Missing bearer token"), 401
        try:
            payload = jwt.decode(header[7:], services().settings.secret_key, algorithms=["HS256"])
        except jwt.PyJWTError:
            return jsonify(error="Invalid or expired token"), 401
        user = services().store.get_user(payload["sub"])
        if user is None:
            return jsonify(error="Unknown user"), 401
        g.user = user
        return view(*args, **kwargs)

    return wrapper
