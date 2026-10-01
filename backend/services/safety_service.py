"""SafetyLayer: the mandatory gate between BioMistral and the user.

Decision order (first match wins):

1. emergency red flag in the query      -> escalate (emergency advice)
2. no usable retrieved context          -> escalate (no_context)
3. too many unsupported claims          -> escalate (hallucination)
4. confidence < CONFIDENCE_THRESHOLD    -> escalate (low_confidence)
5. a few unsupported claims             -> rewrite  (unsupported sentences removed)
6. otherwise                            -> deliver

Confidence is scored on the text that would actually be shown: when a rewrite
is possible, its grounding is that of the rewritten answer (all supported), so
dropping one bad sentence does not by itself force an escalation.

The draft answer is never shown when the action is `escalate`; it is kept in
the result (and in MongoDB) for auditing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from backend.rag.qdrant_store import RetrievedChunk
from backend.safety.confidence import ConfidenceScorer
from backend.safety.emergency import EmergencyClassifier, EmergencyResult
from backend.safety.escalation import EscalationService
from backend.safety.hallucination import HallucinationResult
from backend.safety.triage import RuleTriageClassifier, max_level

DELIVER, REWRITE, ESCALATE = "deliver", "rewrite", "escalate"


@dataclass
class SafetyResult:
    action: str
    triage_level: str
    confidence: float
    hallucination: bool
    emergency: bool
    flags: list[str] = field(default_factory=list)
    reason: str | None = None  # escalation message key
    answer_en: str = ""  # what may be shown to the user (English)
    draft_answer: str | None = None  # raw LLM output, for auditing only
    unsupported_claims: list[str] = field(default_factory=list)
    support_ratio: float | None = None
    confidence_components: dict = field(default_factory=dict)
    emergency_rules: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        return self.action == DELIVER

    @property
    def escalated(self) -> bool:
        return self.action == ESCALATE

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(safe=self.safe, escalated=self.escalated)
        return data


class SafetyLayer:
    def __init__(self, emergency: EmergencyClassifier, triage: RuleTriageClassifier, detector,
                 scorer: ConfidenceScorer, escalation: EscalationService, confidence_threshold: float = 0.7,
                 max_unsupported_ratio_for_rewrite: float = 0.34):
        self.emergency = emergency
        self.triage = triage
        self.detector = detector
        self.scorer = scorer
        self.escalation = escalation
        self.confidence_threshold = confidence_threshold
        self.max_unsupported_ratio = max_unsupported_ratio_for_rewrite

    # -- step 1: runs on every query, before retrieval/generation ------------
    def precheck(self, query_en: str, query_original: str | None = None) -> EmergencyResult:
        return self.emergency.classify(query_en, query_original)

    def classify_triage(self, query_en: str, query_original: str | None = None) -> str:
        return self.triage.classify(query_en, query_original)

    def escalate_emergency(self, emergency: EmergencyResult) -> SafetyResult:
        reason = emergency.message_key or "emergency"
        return SafetyResult(
            action=ESCALATE, triage_level="emergency", confidence=1.0, hallucination=False, emergency=True,
            flags=["emergency"], reason=reason, answer_en=self.escalation.message(reason, "en")[0],
            emergency_rules=emergency.rule_ids,
        )

    def escalate_no_context(self, triage_level: str) -> SafetyResult:
        return SafetyResult(
            action=ESCALATE, triage_level=max_level(triage_level, "gp_appointment"), confidence=0.0,
            hallucination=False, emergency=False, flags=["no_context", "low_confidence"], reason="no_context",
            answer_en=self.escalation.message("no_context", "en")[0],
        )

    # -- step 2: runs on the draft answer --------------------------------------
    def evaluate(self, query_en: str, query_original: str | None, draft: str, chunks: list[RetrievedChunk],
                 mean_token_prob: float | None, emergency: EmergencyResult | None = None) -> SafetyResult:
        emergency = emergency or self.precheck(query_en, query_original)
        if emergency.emergency:
            result = self.escalate_emergency(emergency)
            result.draft_answer = draft
            return result

        triage_level = self.classify_triage(query_en, query_original)
        if not chunks:
            result = self.escalate_no_context(triage_level)
            result.draft_answer = draft
            return result

        check: HallucinationResult = self.detector.check(draft, chunks, query_en)
        rewritable = (check.hallucination and check.supported_claim_count > 0
                      and 1 - check.support_ratio <= self.max_unsupported_ratio)
        # Grounding of what would be shown: a rewrite only keeps supported sentences.
        grounding = 1.0 if rewritable else check.support_ratio
        conf = self.scorer.score([c.score for c in chunks], mean_token_prob, grounding)
        common = dict(
            confidence=conf.score, hallucination=check.hallucination, emergency=False, draft_answer=draft,
            unsupported_claims=check.unsupported_claims, support_ratio=check.support_ratio,
            confidence_components=conf.components,
        )
        flags = ["hallucination"] if check.hallucination else []
        escalated_level = max_level(triage_level, "gp_appointment")

        if check.hallucination and not rewritable:
            return SafetyResult(
                action=ESCALATE, triage_level=escalated_level, flags=flags, reason="hallucination",
                answer_en=self.escalation.message("hallucination", "en")[0], **common,
            )

        if conf.score < self.confidence_threshold:
            return SafetyResult(
                action=ESCALATE, triage_level=escalated_level, flags=["low_confidence", *flags],
                reason="low_confidence", answer_en=self.escalation.message("low_confidence", "en")[0], **common,
            )

        if rewritable:
            rewritten = f"{check.supported_text()}\n\n{self.escalation.rewrite_note}"
            return SafetyResult(action=REWRITE, triage_level=triage_level, flags=flags, answer_en=rewritten, **common)

        return SafetyResult(action=DELIVER, triage_level=triage_level, flags=[], answer_en=draft, **common)
