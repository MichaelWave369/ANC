"""Deterministic runtime anomaly/capture detector producing ThreatSignal records."""

from __future__ import annotations

from anc.core.models import Severity, SignalCategory, SignalLayer, ThreatSignal
from anc.detectors.common import (
    DetectionRule,
    compile_rule,
    dedupe_signals,
    evidence_excerpt,
    make_signal_id,
    normalize_text_blocks,
)

RUNTIME_FACES = ["foundation", "logic", "integrity", "stability"]

ANOMALY_RULES: tuple[DetectionRule, ...] = (
    compile_rule("rm_rf", r"rm\s+-rf", Severity.CRITICAL, 0.96),
    compile_rule("curl_pipe_sh", r"curl\s+[^\n|]*\|\s*sh", Severity.CRITICAL, 0.95),
    compile_rule("powershell_encoded", r"powershell\s+-enc", Severity.HIGH, 0.88),
    compile_rule("chmod_777", r"chmod\s+777", Severity.HIGH, 0.83),
    compile_rule("base64_exec", r"base64\s+(-d|--decode).*(sh|bash|python)", Severity.CRITICAL, 0.91),
    compile_rule("shell_true", r"subprocess\.[a-z_]+\([^\)]*shell\s*=\s*True", Severity.HIGH, 0.85),
    compile_rule("os_system_untrusted", r"os\.system\(|\beval\(|\bexec\(", Severity.HIGH, 0.82),
)

CAPTURE_RULES: tuple[DetectionRule, ...] = (
    compile_rule("disable_logging", r"(disable|turn off).*(logging|audit)", Severity.CRITICAL, 0.92),
    compile_rule("disable_protection", r"disable\s+(protections?|security)", Severity.CRITICAL, 0.93),
)


def _from_rules(
    text: str,
    source_ref: str,
    rules: tuple[DetectionRule, ...],
    category: SignalCategory,
    namespace: str,
) -> list[ThreatSignal]:
    signals: list[ThreatSignal] = []
    for rule in rules:
        for idx, match in enumerate(rule.pattern.finditer(text), start=1):
            signals.append(
                ThreatSignal(
                    signal_id=make_signal_id(namespace, source_ref, rule.name, match.group(0), idx),
                    layer=SignalLayer.RUNTIME,
                    category=category,
                    severity=rule.severity,
                    confidence=rule.confidence,
                    affected_faces=RUNTIME_FACES,
                    evidence_excerpt=evidence_excerpt(text, match.start(), match.end()),
                    source_ref=source_ref,
                    metadata={"rule": rule.name, "match": match.group(0)},
                ),
            )
    return signals


def detect_runtime_signals(commands: list[str] | str, source_ref: str = "") -> list[ThreatSignal]:
    """Return deterministic runtime anomaly/capture signals from command/code strings."""

    signals: list[ThreatSignal] = []
    blocks = normalize_text_blocks(commands)
    for block_idx, text in enumerate(blocks, start=1):
        block_source = f"{source_ref}#run{block_idx}" if source_ref else f"run{block_idx}"
        signals.extend(_from_rules(text, block_source, ANOMALY_RULES, SignalCategory.ANOMALY, "runtime-anom"))
        signals.extend(_from_rules(text, block_source, CAPTURE_RULES, SignalCategory.CAPTURE, "runtime-cap"))
    return dedupe_signals(signals)
