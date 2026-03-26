"""Deterministic posture transition logic for topology-aware ANC security state."""

from __future__ import annotations

from anc.core.models import CoherenceSecurityState, Posture


def determine_posture(state: CoherenceSecurityState) -> Posture:
    """Compute posture using deterministic thresholds on field shape and magnitude."""

    if (
        state.weakest_face
        and (
            state.face_scores[state.weakest_face] < 0.12
            or (state.contamination_load > 0.78 and state.recovery_vertex_score < 0.25)
            or (state.field_average < 0.28 and state.anchor_integrity < 0.35)
        )
    ):
        return Posture.QUARANTINED

    if (
        state.face_scores[state.weakest_face] < 0.25
        or state.contamination_load > 0.62
        or state.field_average < 0.45
        or (state.field_variance > 0.09 and not state.balanced)
        or (state.contamination_load > 0.20 and state.active_signal_count >= 2 and state.anchor_integrity < 0.45)
    ):
        return Posture.HOSTILE

    if (
        state.field_average < 0.65
        or state.field_variance > 0.045
        or state.face_scores[state.weakest_face] < 0.45
        or state.edge_flow_score < 0.55
    ):
        return Posture.DEGRADED

    if (
        state.field_average < 0.82
        or state.face_scores[state.weakest_face] < 0.65
        or state.field_variance > 0.02
        or state.active_signal_count > 0
    ):
        return Posture.WATCH

    return Posture.SAFE
