"""Deterministic ANC governance planning layer (v0.1) for PhiKernel-style substrates.

This module is advisory/planning-only. It does not execute enforcement side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anc.core.incidents import IncidentRecord
from anc.core.models import Action, Posture
from anc.integrations.phikernel_soc_adapter import IntegrationGuardResult


@dataclass(slots=True)
class GovernanceActionPlan:
    """Normalized substrate-facing governance action plan derived from guard results."""

    allow_execution: bool
    require_review: bool
    warn_only: bool
    shadow_execution: bool
    sandbox_execution: bool
    deny_memory_write: bool
    deny_output_commit: bool
    quarantine_branch: bool
    seal_snapshot: bool
    open_incident: bool
    operator_message: str
    rationale: str
    posture: str
    action: str
    related_signal_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def _base_plan(result: IntegrationGuardResult, context: str) -> GovernanceActionPlan:
    action = result.decision.action
    posture = result.state.posture

    plan = GovernanceActionPlan(
        allow_execution=result.allowed,
        require_review=result.requires_review,
        warn_only=False,
        shadow_execution=False,
        sandbox_execution=False,
        deny_memory_write=False,
        deny_output_commit=False,
        quarantine_branch=result.should_quarantine,
        seal_snapshot=result.should_seal,
        open_incident=False,
        operator_message=f"[{context}] {result.summary}",
        rationale=result.decision.rationale,
        posture=posture.value,
        action=action.value,
        related_signal_ids=[signal.signal_id for signal in result.signals],
        metadata={"context": context, "signal_count": len(result.signals)},
    )

    if action == Action.ALLOW:
        pass
    elif action == Action.WARN:
        plan.warn_only = True
        plan.require_review = True
    elif action == Action.SHADOW:
        plan.shadow_execution = True
        plan.require_review = True
    elif action == Action.SANDBOX:
        plan.sandbox_execution = True
        plan.require_review = True
    elif action == Action.REFUSE:
        plan.allow_execution = False
        plan.require_review = True
        plan.open_incident = True
    elif action == Action.QUARANTINE:
        plan.allow_execution = False
        plan.quarantine_branch = True
        plan.open_incident = True
    elif action == Action.SEAL:
        plan.allow_execution = False
        plan.seal_snapshot = True
        plan.quarantine_branch = True
        plan.open_incident = True

    if posture in {Posture.HOSTILE.value, Posture.QUARANTINED.value}:
        plan.open_incident = True

    return plan


def plan_pre_service_governance(result: IntegrationGuardResult) -> GovernanceActionPlan:
    """Plan governance actions before service execution starts."""

    plan = _base_plan(result, context="pre_service")
    # Pre-service is execution gating context; no write/commit deny fields by default.
    return plan


def plan_post_service_governance(result: IntegrationGuardResult) -> GovernanceActionPlan:
    """Plan governance actions after service execution but before finalization."""

    plan = _base_plan(result, context="post_service")
    if not plan.allow_execution or plan.quarantine_branch or plan.seal_snapshot:
        plan.deny_output_commit = True
    return plan


def plan_memory_write_governance(result: IntegrationGuardResult) -> GovernanceActionPlan:
    """Plan governance actions for memory append/write context."""

    plan = _base_plan(result, context="memory_write")
    if not plan.allow_execution or plan.quarantine_branch or plan.seal_snapshot:
        plan.deny_memory_write = True
    return plan


def plan_output_governance(result: IntegrationGuardResult) -> GovernanceActionPlan:
    """Plan governance actions for output commit/display context."""

    plan = _base_plan(result, context="output")
    if not plan.allow_execution or plan.quarantine_branch or plan.seal_snapshot:
        plan.deny_output_commit = True
    return plan


def apply_governance_plan_to_context(plan: GovernanceActionPlan, context: dict[str, Any]) -> dict[str, Any]:
    """Return side-effect-free normalized substrate outcome from a governance plan."""

    if plan.seal_snapshot:
        status = "blocked"
        next_step = "seal_and_quarantine"
    elif plan.quarantine_branch:
        status = "blocked"
        next_step = "quarantine_branch"
    elif not plan.allow_execution:
        status = "blocked"
        next_step = "deny"
    elif plan.sandbox_execution:
        status = "allowed_with_constraints"
        next_step = "sandbox"
    elif plan.shadow_execution:
        status = "allowed_with_constraints"
        next_step = "shadow"
    elif plan.warn_only:
        status = "allowed_with_warning"
        next_step = "warn"
    else:
        status = "allowed"
        next_step = "continue"

    return {
        "status": status,
        "next_step": next_step,
        "operator_message": plan.operator_message,
        "blocked": not plan.allow_execution,
        "allowed": plan.allow_execution,
        "review_required": plan.require_review,
        "quarantine_requested": plan.quarantine_branch,
        "seal_requested": plan.seal_snapshot,
        "context": context,
    }


def build_governance_incident(
    plan: GovernanceActionPlan,
    result: IntegrationGuardResult,
    context_name: str = "",
    incident_id: str = "",
) -> IncidentRecord:
    """Build an IncidentRecord enriched with governance plan metadata."""

    resolved_incident_id = incident_id or f"gov-{result.signals[0].signal_id if result.signals else 'none'}"
    return IncidentRecord(
        incident_id=resolved_incident_id,
        summary=plan.operator_message,
        decision=result.decision,
        related_signal_ids=plan.related_signal_ids,
        before_posture=Posture.SAFE,
        after_posture=result.state.posture,
        metadata={
            "context_name": context_name,
            "plan": {
                "allow_execution": plan.allow_execution,
                "require_review": plan.require_review,
                "warn_only": plan.warn_only,
                "shadow_execution": plan.shadow_execution,
                "sandbox_execution": plan.sandbox_execution,
                "deny_memory_write": plan.deny_memory_write,
                "deny_output_commit": plan.deny_output_commit,
                "quarantine_branch": plan.quarantine_branch,
                "seal_snapshot": plan.seal_snapshot,
                "open_incident": plan.open_incident,
            },
        },
    )
