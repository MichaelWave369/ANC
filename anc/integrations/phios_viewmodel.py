"""PhiOS-facing trust observability view-model layer for ANC (presentation-ready, UI-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from anc.core.incidents import IncidentRecord
from anc.core.models import CoherenceSecurityState, Posture
from anc.integrations.phikernel_enforcement import RuntimeEnforcementResult

POSTURE_LABELS: dict[str, str] = {
    Posture.SAFE.value: "Safe",
    Posture.WATCH.value: "Watch",
    Posture.DEGRADED.value: "Degraded",
    Posture.HOSTILE.value: "Hostile",
    Posture.QUARANTINED.value: "Quarantined",
}

POSTURE_COLORS: dict[str, str] = {
    Posture.SAFE.value: "green",
    Posture.WATCH.value: "amber",
    Posture.DEGRADED.value: "orange",
    Posture.HOSTILE.value: "red",
    Posture.QUARANTINED.value: "purple",
}


@dataclass(slots=True)
class TrustPostureView:
    posture: str
    posture_label: str
    posture_color_hint: str
    recommended_action: str
    operator_message: str
    field_average: float
    field_variance: float
    weakest_face: str
    weakest_face_score: float
    contamination_load: float
    anchor_integrity: float
    recovery_vertex_score: float
    edge_flow_score: float
    balanced: bool
    active_signal_count: int


@dataclass(slots=True)
class FaceHealthView:
    face: str
    score: float
    status: str
    is_weakest: bool


@dataclass(slots=True)
class IncidentView:
    incident_id: str
    timestamp: str
    summary: str
    before_posture: str
    after_posture: str
    action: str
    related_signal_count: int
    operator_severity: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EnforcementView:
    executed: bool
    blocked: bool
    blocked_stage: str | None
    review_required: bool
    quarantined: bool
    sealed: bool
    memory_write_denied: bool
    output_commit_denied: bool
    next_step: str
    action: str
    posture: str
    operator_message: str


@dataclass(slots=True)
class TrustObservatorySnapshot:
    posture: TrustPostureView
    faces: list[FaceHealthView]
    recent_incidents: list[IncidentView]
    enforcement: EnforcementView | None
    summary_cards: dict[str, str]
    topology_flags: list[str]


def _face_status(score: float) -> str:
    if score >= 0.85:
        return "strong"
    if score >= 0.65:
        return "stable"
    if score >= 0.45:
        return "weak"
    return "critical"


def operator_severity(posture: str, action: str = "") -> str:
    if posture in {Posture.HOSTILE.value, Posture.QUARANTINED.value}:
        return "critical"
    if action in {"quarantine", "seal", "refuse"}:
        return "critical"
    if posture == Posture.DEGRADED.value or action in {"sandbox", "shadow"}:
        return "elevated"
    if posture == Posture.WATCH.value or action == "warn":
        return "caution"
    return "nominal"


def build_trust_posture_view(state: CoherenceSecurityState) -> TrustPostureView:
    posture = state.posture.value
    action = state.recommended_action.value
    operator_msg = (
        f"{POSTURE_LABELS.get(posture, posture)} posture; "
        f"recommended action: {action}; weakest face: {state.weakest_face}."
    )
    return TrustPostureView(
        posture=posture,
        posture_label=POSTURE_LABELS.get(posture, posture.title()),
        posture_color_hint=POSTURE_COLORS.get(posture, "gray"),
        recommended_action=action,
        operator_message=operator_msg,
        field_average=state.field_average,
        field_variance=state.field_variance,
        weakest_face=state.weakest_face,
        weakest_face_score=state.face_scores[state.weakest_face],
        contamination_load=state.contamination_load,
        anchor_integrity=state.anchor_integrity,
        recovery_vertex_score=state.recovery_vertex_score,
        edge_flow_score=state.edge_flow_score,
        balanced=state.balanced,
        active_signal_count=state.active_signal_count,
    )


def build_face_health_views(state: CoherenceSecurityState) -> list[FaceHealthView]:
    faces: list[FaceHealthView] = []
    for face, score in state.face_scores.items():
        faces.append(
            FaceHealthView(
                face=face,
                score=score,
                status=_face_status(score),
                is_weakest=face == state.weakest_face,
            ),
        )
    return faces


def build_incident_view(incident: IncidentRecord) -> IncidentView:
    action = incident.decision.action.value if incident.decision else "unknown"
    after_posture = incident.after_posture.value
    return IncidentView(
        incident_id=incident.incident_id,
        timestamp=incident.timestamp.isoformat(),
        summary=incident.summary,
        before_posture=incident.before_posture.value,
        after_posture=after_posture,
        action=action,
        related_signal_count=len(incident.related_signal_ids),
        operator_severity=operator_severity(after_posture, action),
        metadata=incident.metadata,
    )


def build_incident_timeline(incidents: list[IncidentRecord]) -> list[IncidentView]:
    """Build timeline sorted by timestamp descending (most recent first)."""

    ordered = sorted(incidents, key=lambda item: item.timestamp, reverse=True)
    return [build_incident_view(incident) for incident in ordered]


def build_enforcement_view(result: RuntimeEnforcementResult) -> EnforcementView:
    plan = result.governance_plan
    return EnforcementView(
        executed=result.executed,
        blocked=result.blocked,
        blocked_stage=result.blocked_stage,
        review_required=result.review_required,
        quarantined=result.quarantined,
        sealed=result.sealed,
        memory_write_denied=result.memory_write_denied,
        output_commit_denied=result.output_commit_denied,
        next_step=result.next_step,
        action=plan.action,
        posture=plan.posture,
        operator_message=result.operator_message,
    )


def topology_flags_from_state(state: CoherenceSecurityState) -> list[str]:
    flags: list[str] = []
    flags.append(f"Weakest face is {state.weakest_face}")

    if state.balanced:
        flags.append("Balanced field")
    if state.field_variance > 0.045:
        flags.append("Field fragmentation elevated")
    if state.edge_flow_score < 0.55:
        flags.append("Edge transfer degraded")
    if state.recovery_vertex_score < 0.40:
        flags.append("Recovery viability low")
    if state.contamination_load > 0.20:
        flags.append("Contamination pressure elevated")
    if state.face_scores[state.weakest_face] < 0.45:
        flags.append("Localized collapse detected")

    return flags


def _summary_cards(posture: TrustPostureView) -> dict[str, str]:
    return {
        "posture": posture.posture_label,
        "weakest_face": f"{posture.weakest_face} ({posture.weakest_face_score:.2f})",
        "contamination": f"{posture.contamination_load:.2f}",
        "edge_flow": f"{posture.edge_flow_score:.2f}",
        "recovery": f"{posture.recovery_vertex_score:.2f}",
        "action": posture.recommended_action,
    }


def build_trust_observatory_snapshot(
    state: CoherenceSecurityState,
    incidents: list[IncidentRecord] | None = None,
    enforcement: RuntimeEnforcementResult | None = None,
) -> TrustObservatorySnapshot:
    posture_view = build_trust_posture_view(state)
    incident_views = build_incident_timeline(incidents or [])
    enforcement_view = build_enforcement_view(enforcement) if enforcement else None
    return TrustObservatorySnapshot(
        posture=posture_view,
        faces=build_face_health_views(state),
        recent_incidents=incident_views,
        enforcement=enforcement_view,
        summary_cards=_summary_cards(posture_view),
        topology_flags=topology_flags_from_state(state),
    )


def observatory_from_guarded_result(result: RuntimeEnforcementResult) -> TrustObservatorySnapshot:
    integration = result.metadata.get("integration_result")
    if integration is None:
        raise ValueError("RuntimeEnforcementResult metadata missing integration_result")

    incidents: list[IncidentRecord] = [result.incident] if result.incident is not None else []
    return build_trust_observatory_snapshot(
        state=integration.state,
        incidents=incidents,
        enforcement=result,
    )


def observatory_from_state_and_incidents(
    state: CoherenceSecurityState,
    incidents: list[IncidentRecord] | None = None,
) -> TrustObservatorySnapshot:
    return build_trust_observatory_snapshot(state=state, incidents=incidents, enforcement=None)
