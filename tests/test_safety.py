"""Unit tests for the safety layer components and decision engine."""
import pytest

from backend.config.settings import Settings
from backend.rag.qdrant_store import RetrievedChunk
from backend.safety.confidence import ConfidenceScorer
from backend.safety.emergency import EmergencyClassifier
from backend.safety.escalation import EscalationService
from backend.safety.hallucination import LexicalSupportDetector, is_claim
from backend.safety.triage import RuleTriageClassifier
from backend.services.safety_service import SafetyLayer

S = Settings()

FLU = RetrievedChunk("c1", "Common symptoms of the flu include fever, cough, sore throat, runny nose, body aches, "
                           "headache, chills and fatigue.", 0.8, {"source": "TestRef", "title": "Flu"})


@pytest.fixture(scope="module")
def emergency():
    return EmergencyClassifier.from_yaml(S.red_flags_path)


@pytest.fixture(scope="module")
def triage():
    return RuleTriageClassifier.from_yaml(S.triage_rules_path)


def make_layer(threshold=0.7, scorer=None):
    return SafetyLayer(
        EmergencyClassifier.from_yaml(S.red_flags_path), RuleTriageClassifier.from_yaml(S.triage_rules_path),
        LexicalSupportDetector(0.5), scorer or ConfidenceScorer(retrieval_floor=0.3, retrieval_ceiling=0.8),
        EscalationService.from_yaml(S.messages_path), confidence_threshold=threshold,
    )


# --- emergency classifier ---------------------------------------------------

@pytest.mark.parametrize("query", [
    "I have severe chest pain and difficulty breathing.",
    "my father has crushing pain in his chest",
    "I can't breathe properly",
    "I have a severe headache, nausea, and blurred vision. Should I seek immediate medical attention?",
    "My face is drooping and I have slurred speech",
    "my baby has a fever and is not feeding",
    "I took an overdose of sleeping pills",
    "मुझे सीने में दर्द है",
    "నాకు ఛాతీ నొప్పి ఉంది",
])
def test_red_flags_detected(emergency, query):
    assert emergency.classify(query).emergency


@pytest.mark.parametrize("query", [
    "What are the common symptoms of the flu?",
    "What lifestyle changes can help improve cardiovascular health?",
    "I have a mild headache since yesterday",
    "I do not have chest pain, just a cough",
    "No difficulty breathing, only a runny nose",
])
def test_non_emergencies_not_flagged(emergency, query):
    assert not emergency.classify(query).emergency


def test_negation_is_clause_scoped(emergency):
    assert emergency.classify("No fever, but I have chest pain").emergency


def test_self_harm_uses_crisis_message(emergency):
    result = emergency.classify("I want to end my life")
    assert result.emergency and result.message_key == "mental_health_crisis"


# --- triage -------------------------------------------------------------------

@pytest.mark.parametrize("query,level", [
    ("What are the common symptoms of the flu?", "self_care"),
    ("I have had a cough for three weeks", "gp_appointment"),
    ("I have a high fever and can't keep fluids down", "urgent_care"),
    ("There is blood in my urine", "urgent_care"),
])
def test_triage_levels(triage, query, level):
    assert triage.classify(query) == level


# --- confidence --------------------------------------------------------------

def test_confidence_combines_and_renormalises():
    scorer = ConfidenceScorer(0.3, 0.3, 0.4, retrieval_floor=0.3, retrieval_ceiling=0.8)
    full = scorer.score([0.8], 1.0, 1.0)
    assert full.score == pytest.approx(1.0)
    no_gen = scorer.score([0.55], None, 1.0)  # retrieval 0.5, grounding 1.0
    assert set(no_gen.components) == {"retrieval", "grounding"}
    assert no_gen.score == pytest.approx((0.3 * 0.5 + 0.4 * 1.0) / 0.7, abs=1e-3)
    assert scorer.score([], None, None).score == 0.0


# --- hallucination -------------------------------------------------------------

def test_supported_answer_passes():
    result = LexicalSupportDetector(0.5).check("Flu symptoms include fever, cough and body aches.", [FLU])
    assert not result.hallucination and result.support_ratio == 1.0


def test_unsupported_claim_detected():
    answer = "Flu symptoms include fever and cough. Flu is cured permanently by drinking vinegar twice daily."
    result = LexicalSupportDetector(0.5).check(answer, [FLU])
    assert result.hallucination
    assert result.unsupported_claims == ["Flu is cured permanently by drinking vinegar twice daily."]
    assert result.support_ratio == 0.5


def test_advice_sentences_are_not_claims():
    assert not is_claim("Please consult a doctor if the symptoms get worse.")
    assert not is_claim("Seek immediate medical attention.")
    assert is_claim("Metformin lowers blood sugar.")


# --- decision engine ----------------------------------------------------------

def test_deliver_when_everything_passes():
    result = make_layer().evaluate("What are flu symptoms?", None, "Flu symptoms include fever, cough and chills.",
                                   [FLU], 0.9)
    assert result.action == "deliver" and result.safe and not result.flags


def test_emergency_escalates_even_with_good_answer():
    result = make_layer().evaluate("I have chest pain and fever", None, "Flu symptoms include fever.", [FLU], 0.99)
    assert result.action == "escalate" and result.emergency and result.triage_level == "emergency"
    assert "emergency" in result.answer_en.lower()
    assert result.draft_answer == "Flu symptoms include fever."  # kept for audit, not shown


def test_low_confidence_escalates():
    class Fixed(ConfidenceScorer):
        def score(self, *args, **kwargs):
            from backend.safety.confidence import ConfidenceResult
            return ConfidenceResult(0.35, {})

    result = make_layer(scorer=Fixed()).evaluate("What are flu symptoms?", None, "Flu causes fever and cough.",
                                                 [FLU], 0.9)
    assert result.action == "escalate" and result.reason == "low_confidence"
    assert result.confidence == 0.35 and "low_confidence" in result.flags
    assert result.triage_level == "gp_appointment"
    assert "doctor" in result.answer_en


def test_minor_hallucination_is_rewritten():
    answer = ("Flu symptoms include fever and cough. Flu also causes sore throat and body aches. "
              "Common flu signs are headache, chills and fatigue. Flu always lasts exactly forty days.")
    # 1 of 4 claims unsupported: with the default threshold this must be rewritten, not escalated
    result = make_layer(threshold=0.7).evaluate("flu symptoms?", None, answer, [FLU], 0.9)
    assert result.action == "rewrite" and result.hallucination and result.confidence >= 0.7
    assert "forty days" not in result.answer_en and "fever and cough" in result.answer_en
    assert result.unsupported_claims == ["Flu always lasts exactly forty days."]


def test_major_hallucination_escalates():
    answer = "Flu symptoms include fever. Vinegar baths permanently cure influenza. Garlic necklaces prevent viruses."
    result = make_layer(threshold=0.0).evaluate("flu symptoms?", None, answer, [FLU], 0.9)
    assert result.action == "escalate" and result.reason == "hallucination" and result.hallucination


def test_no_context_escalates():
    result = make_layer().evaluate("What is xyzzy syndrome?", None, "It is rare.", [], 0.9)
    assert result.action == "escalate" and result.reason == "no_context"


def test_questions_are_not_claims():
    assert not is_claim("Have you been diagnosed with any specific condition?")


def test_restating_the_patient_is_not_a_hallucination():
    knee = RetrievedChunk("k", "Joint pain can be caused by injury, arthritis or overuse.", 0.7, {})
    answer = "Your knee pain keeps coming back after walking. Joint pain can be caused by injury or arthritis."
    without = LexicalSupportDetector(0.5).check(answer, [knee])
    with_query = LexicalSupportDetector(0.5).check(answer, [knee], query="My knee pain keeps coming back after walking.")
    assert without.hallucination and not with_query.hallucination
