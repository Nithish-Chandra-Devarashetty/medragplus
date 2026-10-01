"""POST /register, POST /login, GET /me"""
from __future__ import annotations

import re

from flask import Blueprint, g, jsonify, request

from backend.api.auth import hash_password, issue_token, public_user, require_auth, services, verify_password
from backend.db.store import DuplicateUserError

bp = Blueprint("auth", __name__)

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@bp.post("/register")
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip() or email.split("@")[0]
    language = data.get("language") or "en"
    if not EMAIL.match(email):
        return jsonify(error="A valid email is required"), 400
    if len(password) < 8:
        return jsonify(error="Password must be at least 8 characters"), 400
    if language not in services().settings.supported_languages:
        return jsonify(error=f"Unsupported language: {language}"), 400
    try:
        user = services().store.create_user(email, hash_password(password), name, language)
    except DuplicateUserError:
        return jsonify(error="An account with this email already exists"), 409
    return jsonify(token=issue_token(user["_id"]), user=public_user(user)), 201


@bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    user = services().store.get_user_by_email((data.get("email") or "").strip())
    if user is None or not verify_password(user["password_hash"], data.get("password") or ""):
        return jsonify(error="Invalid email or password"), 401
    return jsonify(token=issue_token(user["_id"]), user=public_user(user))


@bp.get("/me")
@require_auth
def me():
    return jsonify(user=public_user(g.user))
