"""Deterministic memory contamination detector producing ThreatSignal records."""

from __future__ import annotations

from anc.core.models import Severity, SignalCategory, SignalLayer, ThreatSignal
from anc.detectors.common import (
    compile_rule,
    dedupe_signals,
    evidence_excerpt,
    make_signal_id,
    normalize_text_blocks,
)

CONTAMINATION_FACES = ["memory", "continuity", "integrity", "observer"]

RULES = (
    compile_rule("unauthorized_overwrite", r"unauthorized\s+overwrite", Severity.HIGH, 0.87),
    compile_rule("replace_prior_memory", r"replace\s+prior\s+memory", Severity.CRITICAL, 0.93),
    compile_rule("forget_previous", r"forget\s+previous", Severity.MEDIUM, 0.73),
    compile_rule("rewrite_history", r"rewrite\s+history", Severity.HIGH, 0.86),
    compile_rule("forged_provenance", r"forged\s+provenance", Severity.CRITICAL, 0.94),
    compile_rule("fake_citation", r"fake\s+(citation|source)", Severity.HIGH, 0.84),
    compile_rule("poisoned_append", r"append\s+poisoned\s+instruction", Severity.CRITICAL, 0.92),
)


def detect_contamination_signals(records: list[str] | str, source_ref: str = "") -> list[ThreatSignal]:
    """Return deterministic contamination signals from memory records/chunks."""

    signals: list[ThreatSignal] = []
    blocks = normalize_text_blocks(records)
    for block_idx, text in enumerate(blocks, start=1):
        block_source = f"{source_ref}#mem{block_idx}" if source_ref else f"mem{block_idx}"
        for rule in RULES:
            for idx, match in enumerate(rule.pattern.finditer(text), start=1):
                signals.append(
                    ThreatSignal(
                        signal_id=make_signal_id("contam", block_source, rule.name, match.group(0), idx),
                        layer=SignalLayer.MEMORY,
                        category=SignalCategory.CONTAMINATION,
                        severity=rule.severity,
                        confidence=rule.confidence,
                        affected_faces=CONTAMINATION_FACES,
                        evidence_excerpt=evidence_excerpt(text, match.start(), match.end()),
                        source_ref=block_source,
                        metadata={"rule": rule.name, "match": match.group(0)},
                    ),
                )
    return dedupe_signals(signals)
