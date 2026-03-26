"""Deterministic topology-aware scoring helpers for ANC trust state."""

from __future__ import annotations

from statistics import fmean

from anc.core.decisions import recommend_action
from anc.core.models import (
    Action,
    CoherenceSecurityState,
    FACE_LABELS,
    Posture,
    Severity,
    SignalCategory,
    ThreatSignal,
    normalized_face_map,
)
from anc.core.posture import determine_posture

SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.LOW: 0.05,
    Severity.MEDIUM: 0.10,
    Severity.HIGH: 0.20,
    Severity.CRITICAL: 0.35,
}

CATEGORY_MULTIPLIERS: dict[SignalCategory, float] = {
    SignalCategory.JAILBREAK: 1.00,
    SignalCategory.INJECTION: 1.05,
    SignalCategory.CONTAMINATION: 1.20,
    SignalCategory.ANOMALY: 0.90,
    SignalCategory.EXFIL: 1.10,
    SignalCategory.CAPTURE: 1.15,
    SignalCategory.OTHER: 0.80,
}

ADJACENT_FACE_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (FACE_LABELS[i], FACE_LABELS[(i + 1) % len(FACE_LABELS)])
    for i in range(len(FACE_LABELS))
)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def compute_face_scores(
    base_face_scores: dict[str, float],
    signals: list[ThreatSignal] | None = None,
) -> dict[str, float]:
    """Apply deterministic signal pressure to the base 12-face field."""

    scores = normalized_face_map(base_face_scores)
    for signal in signals or []:
        faces = signal.affected_faces if signal.affected_faces else list(FACE_LABELS)
        impact = (
            SEVERITY_WEIGHTS[signal.severity]
            * CATEGORY_MULTIPLIERS[signal.category]
            * signal.confidence
        )
        distributed_impact = impact / len(faces)
        for face in faces:
            scores[face] = _clamp(scores[face] - distributed_impact)
    return scores


def weakest_face(face_scores: dict[str, float]) -> str:
    """Return face key with minimum score."""

    normalized = normalized_face_map(face_scores)
    return min(normalized, key=normalized.get)


def field_average(face_scores: dict[str, float]) -> float:
    """Mean score of the 12-face field."""

    normalized = normalized_face_map(face_scores)
    return float(fmean(normalized.values()))


def field_variance(face_scores: dict[str, float]) -> float:
    """Population variance across the 12-face field."""

    normalized = normalized_face_map(face_scores)
    avg = field_average(normalized)
    return float(sum((score - avg) ** 2 for score in normalized.values()) / len(normalized))


def is_balanced(face_scores: dict[str, float], variance_threshold: float = 0.02) -> bool:
    """Whether field variance suggests balanced topology."""

    return field_variance(face_scores) <= variance_threshold


def edge_flow_score(face_scores: dict[str, float]) -> float:
    """Measure smooth transfer quality across adjacent faces in [0,1]."""

    normalized = normalized_face_map(face_scores)
    mean_delta = fmean(abs(normalized[a] - normalized[b]) for a, b in ADJACENT_FACE_PAIRS)
    return _clamp(1.0 - float(mean_delta))


def contamination_load(signals: list[ThreatSignal] | None = None) -> float:
    """Aggregate contamination pressure from active signals."""

    if not signals:
        return 0.0

    weighted_sum = 0.0
    for signal in signals:
        base = SEVERITY_WEIGHTS[signal.severity] * signal.confidence
        category_bonus = 1.25 if signal.category == SignalCategory.CONTAMINATION else 1.0
        spread = len(signal.affected_faces) / len(FACE_LABELS) if signal.affected_faces else 1.0
        weighted_sum += base * category_bonus * (0.5 + 0.5 * spread)

    return _clamp(weighted_sum / max(1, len(signals)))


def recovery_vertex_score(
    face_scores: dict[str, float],
    edge_flow: float,
    anchor_integrity: float,
    contamination: float,
) -> float:
    """Estimate recovery viability from weakest-region support + topology transfer."""

    normalized = normalized_face_map(face_scores)
    low_faces_mean = fmean(sorted(normalized.values())[:3])
    score = (0.45 * low_faces_mean) + (0.25 * edge_flow) + (0.30 * anchor_integrity) - (0.20 * contamination)
    return _clamp(float(score))


def compute_security_state(
    base_face_scores: dict[str, float],
    signals: list[ThreatSignal] | None = None,
    anchor_integrity: float = 1.0,
) -> CoherenceSecurityState:
    """Build deterministic trust state from base field and active threat signals."""

    active_signals = signals or []
    scores = compute_face_scores(base_face_scores, active_signals)
    weak_face = weakest_face(scores)
    average = field_average(scores)
    variance = field_variance(scores)
    balanced = is_balanced(scores)
    edge_flow = edge_flow_score(scores)
    contam = contamination_load(active_signals)
    recovery = recovery_vertex_score(scores, edge_flow, _clamp(anchor_integrity), contam)

    provisional = CoherenceSecurityState(
        posture=Posture.SAFE,
        face_scores=scores,
        weakest_face=weak_face,
        field_average=average,
        field_variance=variance,
        balanced=balanced,
        edge_flow_score=edge_flow,
        contamination_load=contam,
        anchor_integrity=_clamp(anchor_integrity),
        recovery_vertex_score=recovery,
        active_signal_count=len(active_signals),
        recommended_action=Action.ALLOW,
    )

    posture = determine_posture(provisional)
    with_posture = CoherenceSecurityState(
        posture=posture,
        face_scores=provisional.face_scores,
        weakest_face=provisional.weakest_face,
        field_average=provisional.field_average,
        field_variance=provisional.field_variance,
        balanced=provisional.balanced,
        edge_flow_score=provisional.edge_flow_score,
        contamination_load=provisional.contamination_load,
        anchor_integrity=provisional.anchor_integrity,
        recovery_vertex_score=provisional.recovery_vertex_score,
        active_signal_count=provisional.active_signal_count,
        recommended_action=Action.ALLOW,
    )

    decision = recommend_action(with_posture)
    with_posture.recommended_action = decision.action
    return with_posture
