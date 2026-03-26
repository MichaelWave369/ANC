"""Deterministic ANC detector layer producing ThreatSignal records for core trust scoring."""

from __future__ import annotations

from anc.core.models import ThreatSignal
from anc.detectors.contamination import detect_contamination_signals
from anc.detectors.injection import detect_injection_signals
from anc.detectors.jailbreak import detect_jailbreak_signals
from anc.detectors.output import detect_output_signals
from anc.detectors.runtime import detect_runtime_signals


def detect_all_input_signals(text: str, source_ref: str = "") -> list[ThreatSignal]:
    """Aggregate deterministic input-layer signals (jailbreak + injection)."""

    return [
        *detect_jailbreak_signals(text, source_ref=source_ref),
        *detect_injection_signals(text, source_ref=source_ref),
    ]


def detect_all_runtime_signals(commands: list[str] | str, source_ref: str = "") -> list[ThreatSignal]:
    """Aggregate deterministic runtime signals."""

    return detect_runtime_signals(commands, source_ref=source_ref)


def detect_all_output_signals(text: str, source_ref: str = "") -> list[ThreatSignal]:
    """Aggregate deterministic output-layer signals."""

    return detect_output_signals(text, source_ref=source_ref)


def detect_all_signals(
    input_text: str = "",
    memory_records: list[str] | None = None,
    runtime_commands: list[str] | None = None,
    output_text: str = "",
    source_ref: str = "",
) -> list[ThreatSignal]:
    """Aggregate detector outputs across input, memory, runtime, and output text surfaces."""

    signals: list[ThreatSignal] = []
    if input_text:
        signals.extend(detect_all_input_signals(input_text, source_ref=source_ref))
    if memory_records:
        signals.extend(detect_contamination_signals(memory_records, source_ref=source_ref))
    if runtime_commands:
        signals.extend(detect_all_runtime_signals(runtime_commands, source_ref=source_ref))
    if output_text:
        signals.extend(detect_all_output_signals(output_text, source_ref=source_ref))
    return signals


__all__ = [
    "detect_all_input_signals",
    "detect_all_output_signals",
    "detect_all_runtime_signals",
    "detect_all_signals",
    "detect_contamination_signals",
    "detect_injection_signals",
    "detect_jailbreak_signals",
    "detect_output_signals",
    "detect_runtime_signals",
]
