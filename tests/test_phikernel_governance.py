from anc.core.models import Action, CoherenceSecurityState, GuardDecision, Posture
from anc.integrations.phikernel_governance import (
    apply_governance_plan_to_context,
    build_governance_incident,
    plan_memory_write_governance,
    plan_output_governance,
    plan_post_service_governance,
    plan_pre_service_governance,
)
from anc.integrations.phikernel_soc_adapter import IntegrationGuardResult


def _result(action: Action, posture: Posture, allowed: bool = True, requires_review: bool = False, should_seal: bool = False, should_quarantine: bool = False) -> IntegrationGuardResult:
    state = CoherenceSecurityState(
        posture=posture,
        face_scores={
            "foundation": 0.8,
            "continuity": 0.8,
            "integrity": 0.8,
            "logic": 0.8,
            "memory": 0.8,
            "recovery": 0.8,
            "stability": 0.8,
            "observer": 0.8,
            "face_09": 0.8,
            "face_10": 0.8,
            "face_11": 0.8,
            "face_12": 0.8,
        },
        weakest_face="foundation",
        field_average=0.8,
        field_variance=0.01,
        balanced=True,
        edge_flow_score=0.8,
        contamination_load=0.1,
        anchor_integrity=0.8,
        recovery_vertex_score=0.8,
        active_signal_count=1,
        recommended_action=action,
    )
    decision = GuardDecision(
        action=action,
        rationale="test rationale",
        triggered_by=["unit"],
        operator_visible=True,
        requires_recovery=action in {Action.SHADOW, Action.SANDBOX, Action.REFUSE, Action.QUARANTINE, Action.SEAL},
    )
    return IntegrationGuardResult(
        state=state,
        decision=decision,
        signals=[],
        allowed=allowed,
        requires_review=requires_review,
        should_seal=should_seal,
        should_quarantine=should_quarantine,
        summary="summary",
    )


def test_safe_allow_pre_service() -> None:
    plan = plan_pre_service_governance(_result(Action.ALLOW, Posture.SAFE, allowed=True))
    assert plan.allow_execution is True
    assert plan.require_review is False
    assert plan.quarantine_branch is False
    assert plan.seal_snapshot is False


def test_watch_warn_pre_service() -> None:
    plan = plan_pre_service_governance(_result(Action.WARN, Posture.WATCH, allowed=True, requires_review=True))
    assert plan.allow_execution is True
    assert plan.require_review is True
    assert plan.warn_only is True


def test_degraded_shadow() -> None:
    plan = plan_pre_service_governance(_result(Action.SHADOW, Posture.DEGRADED, allowed=True, requires_review=True))
    assert plan.allow_execution is True
    assert plan.shadow_execution is True
    assert plan.require_review is True


def test_degraded_sandbox() -> None:
    plan = plan_pre_service_governance(_result(Action.SANDBOX, Posture.DEGRADED, allowed=True, requires_review=True))
    assert plan.allow_execution is True
    assert plan.sandbox_execution is True
    assert plan.require_review is True


def test_hostile_refuse_memory_write_denied() -> None:
    plan = plan_memory_write_governance(_result(Action.REFUSE, Posture.HOSTILE, allowed=False, requires_review=True))
    assert plan.allow_execution is False
    assert plan.deny_memory_write is True
    assert plan.open_incident is True


def test_hostile_quarantine_pre_service() -> None:
    plan = plan_pre_service_governance(
        _result(Action.QUARANTINE, Posture.HOSTILE, allowed=False, should_quarantine=True),
    )
    assert plan.allow_execution is False
    assert plan.quarantine_branch is True


def test_quarantined_seal_output() -> None:
    plan = plan_output_governance(
        _result(Action.SEAL, Posture.QUARANTINED, allowed=False, should_seal=True, should_quarantine=True),
    )
    assert plan.allow_execution is False
    assert plan.deny_output_commit is True
    assert plan.seal_snapshot is True
    assert plan.quarantine_branch is True


def test_context_outcome_interpreter_and_incident() -> None:
    result = _result(Action.QUARANTINE, Posture.HOSTILE, allowed=False, should_quarantine=True)
    plan = plan_post_service_governance(result)
    outcome = apply_governance_plan_to_context(plan, {"ctx": "post_service"})
    incident = build_governance_incident(plan, result, context_name="post_service", incident_id="gov-1")

    assert outcome["status"] == "blocked"
    assert outcome["quarantine_requested"] is True
    assert incident.incident_id == "gov-1"
