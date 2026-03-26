"""Deterministic prompt-injection detector producing ThreatSignal records."""

from __future__ import annotations

from anc.core.models import Severity, SignalCategory, SignalLayer, ThreatSignal
from anc.detectors.common import compile_rule, dedupe_signals, evidence_excerpt, make_signal_id

INJECTION_FACES = ["logic", "integrity", "continuity"]

RULES = (
    compile_rule("ignore_above_below", r"ignore\s+the\s+(above|below)", Severity.MEDIUM, 0.74),
    compile_rule("instead_follow", r"instead\s+follow\s+these\s+instructions", Severity.HIGH, 0.86),
    compile_rule("system_prompt", r"\bsystem\s+prompt\b", Severity.HIGH, 0.81),
    compile_rule("hidden_instruction", r"hidden\s+instruction", Severity.MEDIUM, 0.72),
    compile_rule("override_prior_rules", r"override\s+prior\s+rules", Severity.HIGH, 0.84),
    compile_rule("execute_this_command", r"execute\s+this\s+command", Severity.HIGH, 0.88),
    compile_rule("you_must_now", r"you\s+must\s+now", Severity.MEDIUM, 0.70),
)


def detect_injection_signals(text: str, source_ref: str = "") -> list[ThreatSignal]:
    """Return deterministic injection signals from input/context text."""

    signals: list[ThreatSignal] = []
    for rule in RULES:
        for idx, match in enumerate(rule.pattern.finditer(text), start=1):
            signals.append(
                ThreatSignal(
                    signal_id=make_signal_id("injection", source_ref, rule.name, match.group(0), idx),
                    layer=SignalLayer.INPUT,
                    category=SignalCategory.INJECTION,
                    severity=rule.severity,
                    confidence=rule.confidence,
                    affected_faces=INJECTION_FACES,
                    evidence_excerpt=evidence_excerpt(text, match.start(), match.end()),
                    source_ref=source_ref,
                    metadata={"rule": rule.name, "match": match.group(0)},
                ),
            )
    return dedupe_signals(signals)
