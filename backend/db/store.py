"""MongoDB persistence: users, sessions, messages and safety_flags."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError


class DuplicateUserError(Exception):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:  # pymongo returns naive UTC datetimes unless tz_aware=True
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def connect(settings):
    if settings.mongo_uri.startswith("mongomock://"):
        import mongomock

        client = mongomock.MongoClient()
    else:
        from pymongo import MongoClient

        client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
    return client[settings.mongo_db]


class ConversationStore:
    def __init__(self, db):
        self.db = db
        self.users = db["users"]
        self.sessions = db["sessions"]
        self.messages = db["messages"]
        self.safety_flags = db["safety_flags"]
        self.users.create_index([("email", ASCENDING)], unique=True)
        self.sessions.create_index([("user_id", ASCENDING), ("updated_at", DESCENDING)])
        self.messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
        self.safety_flags.create_index([("created_at", DESCENDING)])

    # -- users -------------------------------------------------------------
    def create_user(self, email: str, password_hash: str, name: str, language: str) -> dict:
        user = {"_id": uuid.uuid4().hex, "email": email.lower(), "password_hash": password_hash, "name": name,
                "language": language, "created_at": utcnow()}
        try:
            self.users.insert_one(user)
        except DuplicateKeyError as exc:
            raise DuplicateUserError(email) from exc
        return user

    def get_user_by_email(self, email: str) -> dict | None:
        return self.users.find_one({"email": email.lower()})

    def get_user(self, user_id: str) -> dict | None:
        return self.users.find_one({"_id": user_id})

    # -- sessions ----------------------------------------------------------
    def create_session(self, user_id: str, title: str, language: str) -> str:
        now = utcnow()
        session_id = uuid.uuid4().hex
        self.sessions.insert_one({"_id": session_id, "user_id": user_id, "title": title[:80], "language": language,
                                  "created_at": now, "updated_at": now, "message_count": 0})
        return session_id

    def get_session(self, session_id: str, user_id: str) -> dict | None:
        return self.sessions.find_one({"_id": session_id, "user_id": user_id})

    def list_sessions(self, user_id: str, limit: int = 100) -> list[dict]:
        return list(self.sessions.find({"user_id": user_id}).sort("updated_at", DESCENDING).limit(limit))

    def delete_session(self, session_id: str, user_id: str) -> bool:
        deleted = self.sessions.delete_one({"_id": session_id, "user_id": user_id}).deleted_count
        if deleted:
            self.messages.delete_many({"session_id": session_id})
            self.safety_flags.delete_many({"session_id": session_id})
        return bool(deleted)

    # -- messages ----------------------------------------------------------
    def add_message(self, message: dict) -> dict:
        message = {"_id": uuid.uuid4().hex, "created_at": utcnow(), **message}
        self.messages.insert_one(message)
        self.sessions.update_one(
            {"_id": message["session_id"]},
            {"$set": {"updated_at": message["created_at"], "language": message.get("response_language")},
             "$inc": {"message_count": 1}},
        )
        return message

    def get_messages(self, session_id: str) -> list[dict]:
        return list(self.messages.find({"session_id": session_id}).sort("created_at", ASCENDING))

    def recent_turns(self, session_id: str, n: int) -> list[dict]:
        """Last n turns, oldest first, in English (the pipeline's working language)."""
        if n <= 0:
            return []
        docs = list(self.messages.find({"session_id": session_id}).sort("created_at", DESCENDING).limit(n))
        return [{"user": d["query_english"], "assistant": d["answer_english"]} for d in reversed(docs)]

    def add_safety_flag(self, flag: dict) -> None:
        self.safety_flags.insert_one({"_id": uuid.uuid4().hex, "created_at": utcnow(), **flag})
