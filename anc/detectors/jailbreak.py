"""Deterministic jailbreak phrase detector producing ThreatSignal records."""

from __future__ import annotations

from anc.core.models import Severity, SignalCategory, SignalLayer, ThreatSignal
from anc.detectors.common import compile_rule, dedupe_signals, evidence_excerpt, make_signal_id

JAILBREAK_FACES = ["logic", "integrity", "observer", "foundation"]

RULES = (
    compile_rule("ignore_previous", r"ignore\s+previous\s+instructions", Severity.HIGH, 0.84),
    compile_rule("bypass_safety", r"bypass\s+safety", Severity.CRITICAL, 0.93),
    compile_rule("do_not_follow_policy", r"do\s+not\s+follow\s+policy", Severity.HIGH, 0.88),
    compile_rule("act_unrestricted", r"act\s+as\s+if\s+unrestricted", Severity.HIGH, 0.86),
    compile_rule("developer_mode", r"\bdeveloper\s+mode\b|\bDAN\b", Severity.HIGH, 0.83),
    compile_rule("reveal_system_prompt", r"reveal\s+(hidden|system)\s+prompt", Severity.CRITICAL, 0.95),
    compile_rule("pretend_rules_absent", r"pretend\s+the\s+rules\s+do\s+not\s+exist", Severity.CRITICAL, 0.90),
)


def detect_jailbreak_signals(text: str, source_ref: str = "") -> list[ThreatSignal]:
    """Return deterministic jailbreak signals from input text."""

    signals: list[ThreatSignal] = []
    for rule in RULES:
        for idx, match in enumerate(rule.pattern.finditer(text), start=1):
            excerpt = evidence_excerpt(text, match.start(), match.end())
            signals.append(
                ThreatSignal(
                    signal_id=make_signal_id("jailbreak", source_ref, rule.name, match.group(0), idx),
                    layer=SignalLayer.INPUT,
                    category=SignalCategory.JAILBREAK,
                    severity=rule.severity,
                    confidence=rule.confidence,
                    affected_faces=JAILBREAK_FACES,
                    evidence_excerpt=excerpt,
                    source_ref=source_ref,
                    metadata={"rule": rule.name, "match": match.group(0)},
                ),
            )
    return dedupe_signals(signals)
