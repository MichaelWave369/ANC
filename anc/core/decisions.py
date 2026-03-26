"""Deterministic guard action recommendations from posture and topology metrics."""

from __future__ import annotations

from anc.core.models import Action, CoherenceSecurityState, GuardDecision, Posture


def recommend_action(state: CoherenceSecurityState) -> GuardDecision:
    """Map posture and risk shape metrics to a deterministic guard decision."""

    triggered_by = [
        f"posture={state.posture.value}",
        f"weakest_face={state.weakest_face}",
        f"contamination={state.contamination_load:.3f}",
    ]

    if state.posture == Posture.SAFE:
        return GuardDecision(
            action=Action.ALLOW,
            rationale="Field is balanced with strong average and low contamination.",
            triggered_by=triggered_by,
            operator_visible=False,
            requires_recovery=False,
        )

    if state.posture == Posture.WATCH:
        return GuardDecision(
            action=Action.WARN,
            rationale="Localized or moderate anomalies detected; operation may continue with caution.",
            triggered_by=triggered_by,
            operator_visible=True,
            requires_recovery=False,
        )

    if state.posture == Posture.DEGRADED:
        action = Action.SHADOW if state.recovery_vertex_score >= 0.45 else Action.SANDBOX
        return GuardDecision(
            action=action,
            rationale="Field fragmentation or face weakness requires constrained execution.",
            triggered_by=triggered_by,
            operator_visible=True,
            requires_recovery=True,
        )

    if state.posture == Posture.HOSTILE:
        action = Action.QUARANTINE if state.contamination_load >= 0.70 else Action.REFUSE
        return GuardDecision(
            action=action,
            rationale="High-confidence hostile posture; isolate or refuse risky operations.",
            triggered_by=triggered_by,
            operator_visible=True,
            requires_recovery=True,
        )

    return GuardDecision(
        action=Action.SEAL,
        rationale="Critical collapse / unrecoverable posture; seal channels pending trusted recovery.",
        triggered_by=triggered_by,
        operator_visible=True,
        requires_recovery=True,
    )
