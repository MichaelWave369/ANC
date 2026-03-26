"""Deterministic output leakage detector producing ThreatSignal records."""

from __future__ import annotations

from anc.core.models import Severity, SignalCategory, SignalLayer, ThreatSignal
from anc.detectors.common import compile_rule, dedupe_signals, evidence_excerpt, make_signal_id

OUTPUT_FACES = ["observer", "integrity", "recovery", "continuity"]

EXFIL_RULES = (
    compile_rule("hidden_prompt_reveal", r"here\s+is\s+the\s+hidden\s+prompt", Severity.CRITICAL, 0.94),
    compile_rule("reveal_system_instruction", r"(revealing|reveal).*(system|policy)\s+instructions?", Severity.HIGH, 0.86),
    compile_rule("token_leak", r"(api[_-]?key|secret[_-]?token|access[_-]?token)\s*[:=]", Severity.CRITICAL, 0.95),
    compile_rule("exfil_hint", r"exfil(trate|tration)|encoded\s+payload", Severity.HIGH, 0.84),
)

JAILBREAK_STYLE_RULES = (
    compile_rule("dangerous_cmd_sequence", r"rm\s+-rf|curl\s+[^\n|]*\|\s*sh", Severity.HIGH, 0.82),
)


def detect_output_signals(text: str, source_ref: str = "") -> list[ThreatSignal]:
    """Return deterministic exfil/jailbreak-like signals from model output text."""

    signals: list[ThreatSignal] = []
    for rule in EXFIL_RULES:
        for idx, match in enumerate(rule.pattern.finditer(text), start=1):
            signals.append(
                ThreatSignal(
                    signal_id=make_signal_id("output-exfil", source_ref, rule.name, match.group(0), idx),
                    layer=SignalLayer.OUTPUT,
                    category=SignalCategory.EXFIL,
                    severity=rule.severity,
                    confidence=rule.confidence,
                    affected_faces=OUTPUT_FACES,
                    evidence_excerpt=evidence_excerpt(text, match.start(), match.end()),
                    source_ref=source_ref,
                    metadata={"rule": rule.name, "match": match.group(0)},
                ),
            )
    for rule in JAILBREAK_STYLE_RULES:
        for idx, match in enumerate(rule.pattern.finditer(text), start=1):
            signals.append(
                ThreatSignal(
                    signal_id=make_signal_id("output-jb", source_ref, rule.name, match.group(0), idx),
                    layer=SignalLayer.OUTPUT,
                    category=SignalCategory.JAILBREAK,
                    severity=rule.severity,
                    confidence=rule.confidence,
                    affected_faces=OUTPUT_FACES,
                    evidence_excerpt=evidence_excerpt(text, match.start(), match.end()),
                    source_ref=source_ref,
                    metadata={"rule": rule.name, "match": match.group(0)},
                ),
            )
    return dedupe_signals(signals)
