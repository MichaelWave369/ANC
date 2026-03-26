"""Deterministic ANC-to-substrate adapter for PhiKernel SoC style guard calls."""

from __future__ import annotations

from dataclasses import dataclass

from anc.core.decisions import recommend_action
from anc.core.incidents import IncidentRecord
from anc.core.models import Action, CoherenceSecurityState, GuardDecision, Posture, ThreatSignal
from anc.core.scoring import compute_security_state
from anc.detectors import (
    detect_all_input_signals,
    detect_all_output_signals,
    detect_all_runtime_signals,
    detect_contamination_signals,
)

DEFAULT_FACE_BASELINE: dict[str, float] = {
    "foundation": 0.85,
    "continuity": 0.85,
    "integrity": 0.85,
    "logic": 0.85,
    "memory": 0.85,
    "recovery": 0.85,
    "stability": 0.85,
    "observer": 0.85,
    "face_09": 0.85,
    "face_10": 0.85,
    "face_11": 0.85,
    "face_12": 0.85,
}


@dataclass(slots=True)
class IntegrationGuardResult:
    """Normalized substrate-facing result from detector + scoring + decision flow."""

    state: CoherenceSecurityState
    decision: GuardDecision
    signals: list[ThreatSignal]
    allowed: bool
    requires_review: bool
    should_seal: bool
    should_quarantine: bool
    summary: str


@dataclass(slots=True)
class PolicyInterpretation:
    """Simple policy interpretation flags derived from guard decision action."""

    allowed: bool
    requires_review: bool
    should_seal: bool
    should_quarantine: bool


def _interpret_action(action: Action) -> PolicyInterpretation:
    if action == Action.ALLOW:
        return PolicyInterpretation(True, False, False, False)
    if action in {Action.WARN, Action.SHADOW, Action.SANDBOX}:
        return PolicyInterpretation(True, True, False, False)
    if action == Action.REFUSE:
        return PolicyInterpretation(False, False, False, False)
    if action == Action.QUARANTINE:
        return PolicyInterpretation(False, False, False, True)
    if action == Action.SEAL:
        return PolicyInterpretation(False, False, True, False)
    return PolicyInterpretation(False, True, False, False)


def _build_result(signals: list[ThreatSignal], source_ref: str, face_scores: dict[str, float], anchor_integrity: float) -> IntegrationGuardResult:
    state = compute_security_state(base_face_scores=face_scores, signals=signals, anchor_integrity=anchor_integrity)
    decision = recommend_action(state)
    flags = _interpret_action(decision.action)
    summary = (
        f"source={source_ref or 'n/a'} posture={state.posture.value} "
        f"action={decision.action.value} signals={len(signals)}"
    )
    return IntegrationGuardResult(
        state=state,
        decision=decision,
        signals=signals,
        allowed=flags.allowed,
        requires_review=flags.requires_review,
        should_seal=flags.should_seal,
        should_quarantine=flags.should_quarantine,
        summary=summary,
    )


def guard_input_text(
    text: str,
    source_ref: str = "",
    memory_records: list[str] | None = None,
    runtime_commands: list[str] | str | None = None,
    base_face_scores: dict[str, float] | None = None,
    anchor_integrity: float = 0.90,
) -> IntegrationGuardResult:
    """Run input-oriented detectors and return normalized integration result."""

    signals = detect_all_input_signals(text, source_ref=source_ref)
    if memory_records:
        signals.extend(detect_contamination_signals(memory_records, source_ref=source_ref))
    if runtime_commands:
        signals.extend(detect_all_runtime_signals(runtime_commands, source_ref=source_ref))
    return _build_result(signals, source_ref, base_face_scores or DEFAULT_FACE_BASELINE, anchor_integrity)


def guard_runtime_commands(
    commands: list[str] | str,
    source_ref: str = "",
    base_face_scores: dict[str, float] | None = None,
    anchor_integrity: float = 0.90,
) -> IntegrationGuardResult:
    """Run runtime detector family and return normalized integration result."""

    signals = detect_all_runtime_signals(commands, source_ref=source_ref)
    return _build_result(signals, source_ref, base_face_scores or DEFAULT_FACE_BASELINE, anchor_integrity)


def guard_memory_records(
    records: list[str] | str,
    source_ref: str = "",
    base_face_scores: dict[str, float] | None = None,
    anchor_integrity: float = 0.90,
) -> IntegrationGuardResult:
    """Run memory contamination detector and return normalized integration result."""

    signals = detect_contamination_signals(records, source_ref=source_ref)
    return _build_result(signals, source_ref, base_face_scores or DEFAULT_FACE_BASELINE, anchor_integrity)


def guard_output_text(
    text: str,
    source_ref: str = "",
    emitted_commands: list[str] | str | None = None,
    base_face_scores: dict[str, float] | None = None,
    anchor_integrity: float = 0.90,
) -> IntegrationGuardResult:
    """Run output detectors and optional runtime checks against emitted command text."""

    signals = detect_all_output_signals(text, source_ref=source_ref)
    if emitted_commands:
        signals.extend(detect_all_runtime_signals(emitted_commands, source_ref=source_ref))
    return _build_result(signals, source_ref, base_face_scores or DEFAULT_FACE_BASELINE, anchor_integrity)


def build_incident_from_result(
    result: IntegrationGuardResult,
    incident_id: str,
    before_posture: Posture = Posture.SAFE,
    summary: str = "",
) -> IncidentRecord:
    """Build an IncidentRecord-ready object from integration guard output."""

    return IncidentRecord(
        incident_id=incident_id,
        summary=summary or result.summary,
        decision=result.decision,
        related_signal_ids=[signal.signal_id for signal in result.signals],
        before_posture=before_posture,
        after_posture=result.state.posture,
        metadata={
            "allowed": result.allowed,
            "requires_review": result.requires_review,
            "should_seal": result.should_seal,
            "should_quarantine": result.should_quarantine,
        },
    )
