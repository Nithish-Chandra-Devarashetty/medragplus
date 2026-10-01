"""End-to-end tests through the REST API (the 8 scenarios from the MedRAG+ spec)."""
import base64
import io

from backend.models.biomistral import FakeLLM
from backend.safety.confidence import ConfidenceResult
from backend.services.speech_service import FakeASR


def chat(client, headers, message, **extra):
    resp = client.post("/api/chat", json={"message": message, **extra}, headers=headers)
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


# --- auth ------------------------------------------------------------------------

def test_register_login_and_auth_required(client, auth_headers):
    assert client.post("/api/register", json={"email": "patient@example.com", "password": "password123"}).status_code == 409
    assert client.post("/api/register", json={"email": "bad", "password": "password123"}).status_code == 400
    assert client.post("/api/login", json={"email": "patient@example.com", "password": "wrong-pass"}).status_code == 401
    login = client.post("/api/login", json={"email": "PATIENT@example.com", "password": "password123"})
    assert login.status_code == 200 and login.get_json()["user"]["email"] == "patient@example.com"
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/api/me", headers=auth_headers).get_json()["user"]["name"] == "Test"


# --- 1. normal English query ---------------------------------------------------

def test_normal_english_query(client, auth_headers):
    data = chat(client, auth_headers, "What are the common symptoms of the flu?")
    assert data["action"] == "deliver" and not data["escalated"] and not data["emergency"]
    assert "fever" in data["answer"].lower()
    assert data["sources"] and data["sources"][0]["title"] == "Influenza (Flu)"
    assert data["sources"][0]["url"].startswith("https://")
    assert data["triage_level"] == "self_care"
    assert 0 <= data["confidence"] <= 1
    assert data["language"] == "en" and data["audio"] is None
    for stage in ("translate_in", "embedding", "retrieval", "generation", "safety", "translate_out", "total"):
        assert stage in data["timings_ms"]


# --- 2 & 3. Hindi and Telugu ----------------------------------------------------

def test_hindi_query_round_trip(client, auth_headers):
    data = chat(client, auth_headers, "फ्लू के सामान्य लक्षण क्या हैं?", language="auto", tts=True)
    assert data["query"]["detected_language"] == "hi"
    assert data["query"]["english"] == "What are the common symptoms of the flu?"
    assert data["language"] == "hi"
    assert data["answer"].startswith("<hi> ")  # translated back by the (fake) NLLB
    assert "fever" in data["answer_english"].lower()
    assert data["audio"]["mime"] == "audio/mpeg"
    assert base64.b64decode(data["audio"]["base64"]).startswith(b"ID3hi:")
    assert "tts" in data["timings_ms"]


def test_telugu_query_round_trip(client, auth_headers):
    data = chat(client, auth_headers, "ఫ్లూ యొక్క సాధారణ లక్షణాలు ఏమిటి?", tts=True)
    assert data["query"]["detected_language"] == "te" and data["language"] == "te"
    assert data["action"] == "deliver" and data["answer"].startswith("<te> ")
    assert base64.b64decode(data["audio"]["base64"]).startswith(b"ID3te:")


# --- 4. emergency ------------------------------------------------------------------

def test_emergency_skips_generation(client, auth_headers, services):
    data = chat(client, auth_headers, "I have severe chest pain and difficulty breathing.")
    assert data["emergency"] and data["escalated"] and data["triage_level"] == "emergency"
    assert "108" in data["answer"]
    assert services.llm.prompts == []  # BioMistral never ran
    assert "generation" not in data["timings_ms"]
    flags = list(services.store.safety_flags.find())
    assert flags and flags[0]["flags"] == ["emergency"]


def test_emergency_in_hindi_uses_pretranslated_message(client, auth_headers):
    data = chat(client, auth_headers, "मुझे सीने में दर्द है")
    assert data["emergency"] and data["language"] == "hi"
    assert "108" in data["answer"] and not data["answer"].startswith("<hi>")  # fixed text, not MT


# --- 5. low confidence -------------------------------------------------------------

def test_low_confidence_escalates(client, auth_headers, services):
    class Fixed:
        def score(self, *args, **kwargs):
            return ConfidenceResult(0.35, {})

    services.safety.scorer = Fixed()
    data = chat(client, auth_headers, "What are the common symptoms of the flu?")
    assert data["escalated"] and data["action"] == "escalate" and data["confidence"] == 0.35
    assert "low_confidence" in data["safety_flags"]
    assert "fever" not in data["answer"].lower()  # the normal answer is withheld
    stored = services.store.messages.find_one({"_id": data["message_id"]})
    assert "fever" in stored["draft_answer"].lower()  # but kept for audit


def test_unknown_topic_has_no_context(client, auth_headers, services):
    services.retrieval.retriever.min_score = 0.99
    data = chat(client, auth_headers, "Tell me about quantum chromodynamics")
    assert data["escalated"] and "no_context" in data["safety_flags"] and services.llm.prompts == []


# --- 6. hallucination ---------------------------------------------------------------

def test_hallucination_is_not_delivered(client, auth_headers, services):
    services.generation.llm = FakeLLM(lambda prompt: "Flu symptoms include fever and cough. "
                                                     "Drinking vinegar permanently cures influenza overnight. "
                                                     "Garlic necklaces block all viruses.")
    data = chat(client, auth_headers, "What are the common symptoms of the flu?")
    assert data["hallucination"]
    assert data["action"] in {"rewrite", "escalate"}
    assert "vinegar" not in data["answer"].lower()
    assert "Drinking vinegar permanently cures influenza overnight." in data["unsupported_claims"]


# --- 7. voice -------------------------------------------------------------------------

def test_voice_flow(client, auth_headers, services):
    services.speech.asr = FakeASR("What are the common symptoms of the flu?", "en")
    resp = client.post("/api/chat/voice", headers=auth_headers, content_type="multipart/form-data",
                       data={"audio": (io.BytesIO(b"fake-webm-bytes"), "rec.webm"), "language": "en", "tts": "true"})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["query"]["transcript"] == "What are the common symptoms of the flu?"
    assert "asr" in data["timings_ms"] and "tts" in data["timings_ms"]
    assert data["audio"] and "fever" in data["answer"].lower()


def test_voice_requires_audio(client, auth_headers):
    resp = client.post("/api/chat/voice", headers=auth_headers, content_type="multipart/form-data", data={})
    assert resp.status_code == 400


def test_tts_endpoint(client, auth_headers):
    resp = client.post("/api/tts", json={"text": "Rest well", "language": "te"}, headers=auth_headers)
    assert resp.status_code == 200 and resp.mimetype == "audio/mpeg" and resp.data == "ID3te:Rest well".encode()


# --- 8. conversation history ----------------------------------------------------------

def test_conversation_history_is_preserved(client, auth_headers, services):
    first = chat(client, auth_headers, "I have a headache.")
    second = chat(client, auth_headers, "It started yesterday.", session_id=first["session_id"])
    assert second["session_id"] == first["session_id"]
    assert second["retrieval_query"] == "I have a headache. It started yesterday."
    last_prompt = services.llm.prompts[-1]
    assert "Patient: I have a headache." in last_prompt  # the LLM sees the earlier turn

    sessions = client.get("/api/history", headers=auth_headers).get_json()["sessions"]
    assert len(sessions) == 1 and sessions[0]["message_count"] == 2 and sessions[0]["title"] == "I have a headache."
    turns = client.get(f"/api/history/{first['session_id']}", headers=auth_headers).get_json()["turns"]
    assert [t["query"]["original"] for t in turns] == ["I have a headache.", "It started yesterday."]


def test_sessions_are_private_and_deletable(client, auth_headers):
    session_id = chat(client, auth_headers, "What are the common symptoms of the flu?")["session_id"]
    other = client.post("/api/register", json={"email": "other@example.com", "password": "password123"}).get_json()
    other_headers = {"Authorization": f"Bearer {other['token']}"}
    assert client.get(f"/api/history/{session_id}", headers=other_headers).status_code == 404
    assert client.post("/api/chat", json={"message": "hi there", "session_id": session_id},
                       headers=other_headers).status_code == 404
    assert client.delete(f"/api/history/{session_id}", headers=auth_headers).get_json() == {"deleted": True}
    assert client.get("/api/history", headers=auth_headers).get_json()["sessions"] == []


def test_health(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok" and data["components"]["knowledge_chunks"] == 5


def test_topic_change_is_not_drowned_by_history(client, auth_headers, services):
    first = chat(client, auth_headers, "I have a headache.")
    second = chat(client, auth_headers, "What are the common symptoms of the flu?", session_id=first["session_id"])
    assert second["sources"][0]["title"] == "Influenza (Flu)"  # new topic wins over the previous turn
    prompt = services.llm.prompts[-1]
    assert "Patient: I have a headache." in prompt
    assert "Assistant:" not in prompt  # earlier replies are not fed back (greedy decoding would copy them)


def test_voice_on_auto_uses_profile_language_as_hint(client, services):
    token = client.post("/api/register", json={"email": "te@example.com", "password": "password123",
                                               "language": "te"}).get_json()["token"]
    hints = []

    class RecordingASR(FakeASR):
        def transcribe(self, audio, language_hint):
            hints.append(language_hint)
            return super().transcribe(audio, language_hint)

    services.speech.asr = RecordingASR("What are the common symptoms of the flu?", None)
    resp = client.post("/api/chat/voice", headers={"Authorization": f"Bearer {token}"},
                       content_type="multipart/form-data",
                       data={"audio": (io.BytesIO(b"x"), "rec.webm"), "language": "auto"})
    assert resp.status_code == 200 and hints == ["te"]
