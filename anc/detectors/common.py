"""Shared deterministic helpers for ANC detector rule matching and signal normalization."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Pattern

from anc.core.models import Severity, ThreatSignal


@dataclass(frozen=True)
class DetectionRule:
    """Single deterministic detection rule for phrase/regex matching."""

    name: str
    pattern: Pattern[str]
    severity: Severity
    confidence: float


def compile_rule(name: str, phrase_or_regex: str, severity: Severity, confidence: float) -> DetectionRule:
    """Compile a case-insensitive regex rule with deterministic metadata."""

    return DetectionRule(
        name=name,
        pattern=re.compile(phrase_or_regex, re.IGNORECASE),
        severity=severity,
        confidence=max(0.0, min(1.0, confidence)),
    )


def make_signal_id(namespace: str, source_ref: str, rule_name: str, match_text: str, index: int) -> str:
    """Create deterministic signal id for stable testability."""

    payload = f"{namespace}|{source_ref}|{rule_name}|{match_text.lower()}|{index}".encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"{namespace}-{digest}"


def normalize_text_blocks(value: list[str] | str) -> list[str]:
    """Normalize a detector input into a deterministic list of text blocks."""

    if isinstance(value, str):
        return [value]
    return [block for block in value if block]


def evidence_excerpt(text: str, start: int, end: int, margin: int = 36) -> str:
    """Extract compact evidence snippet around a regex match."""

    left = max(0, start - margin)
    right = min(len(text), end + margin)
    excerpt = text[left:right].strip().replace("\n", " ")
    return excerpt[:220]


def dedupe_signals(signals: Iterable[ThreatSignal]) -> list[ThreatSignal]:
    """Drop duplicate signals by stable semantic key while preserving order."""

    deduped: list[ThreatSignal] = []
    seen: set[tuple[str, str, str, str]] = set()
    for signal in signals:
        key = (
            signal.layer.value,
            signal.category.value,
            signal.evidence_excerpt.lower(),
            signal.source_ref,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(signal)
    return deduped
