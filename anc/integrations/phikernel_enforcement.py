"""Deterministic local enforcement interpretation layer for PhiKernel-style substrates.

This v0.1 module does not perform external side effects (no sandbox launch/process control).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anc.core.incidents import IncidentRecord
from anc.integrations.phikernel_governance import (
    GovernanceActionPlan,
    build_governance_incident,
    plan_memory_write_governance,
    plan_output_governance,
    plan_post_service_governance,
    plan_pre_service_governance,
)
from anc.integrations.phikernel_hooks import (
    memory_write_guard,
    output_guard,
    post_service_run_guard,
    pre_service_run_guard,
)


@dataclass(slots=True)
class RuntimeEnforcementResult:
    """Runtime-facing outcome after governance plan interpretation."""

    executed: bool
    blocked: bool
    blocked_stage: str | None
    review_required: bool
    quarantined: bool
    sealed: bool
    memory_write_denied: bool
    output_commit_denied: bool
    next_step: str
    operator_message: str
    incident: IncidentRecord | None
    governance_plan: GovernanceActionPlan
    metadata: dict[str, Any] = field(default_factory=dict)


def _enforce_plan(
    plan: GovernanceActionPlan,
    stage: str,
    context: dict[str, Any],
    execution_already_occurred: bool = False,
) -> RuntimeEnforcementResult:
    blocked = False
    blocked_stage: str | None = None
    executed = False
    next_step = "proceed"
    metadata: dict[str, Any] = {"stage": stage}

    if plan.seal_snapshot:
        blocked = True
        blocked_stage = stage
        next_step = "seal_and_review"
    elif plan.quarantine_branch:
        blocked = True
        blocked_stage = stage
        next_step = "quarantine_branch"
    elif plan.deny_memory_write:
        blocked = True
        blocked_stage = stage
        next_step = "deny_memory_write"
    elif plan.deny_output_commit:
        blocked = True
        blocked_stage = stage
        next_step = "deny_output_commit"
    elif not plan.allow_execution:
        blocked = True
        blocked_stage = stage
        next_step = "deny_execution"
    elif plan.sandbox_execution:
        executed = True
        next_step = "proceed_sandbox"
        metadata["execution_mode"] = "sandbox"
    elif plan.shadow_execution:
        executed = True
        next_step = "proceed_shadow"
        metadata["execution_mode"] = "shadow"
    elif plan.warn_only or plan.require_review:
        executed = True
        next_step = "proceed_with_review"
    else:
        executed = True
        next_step = "proceed"

    if execution_already_occurred:
        executed = True

    return RuntimeEnforcementResult(
        executed=executed,
        blocked=blocked,
        blocked_stage=blocked_stage,
        review_required=plan.require_review,
        quarantined=plan.quarantine_branch,
        sealed=plan.seal_snapshot,
        memory_write_denied=plan.deny_memory_write,
        output_commit_denied=plan.deny_output_commit,
        next_step=next_step,
        operator_message=plan.operator_message,
        incident=None,
        governance_plan=plan,
        metadata={**metadata, "context": context},
    )


def enforce_pre_service_plan(plan: GovernanceActionPlan, context: dict[str, Any]) -> RuntimeEnforcementResult:
    return _enforce_plan(plan, stage="pre_service", context=context)


def enforce_post_service_plan(plan: GovernanceActionPlan, context: dict[str, Any]) -> RuntimeEnforcementResult:
    return _enforce_plan(plan, stage="post_service", context=context, execution_already_occurred=True)


def enforce_memory_write_plan(plan: GovernanceActionPlan, context: dict[str, Any]) -> RuntimeEnforcementResult:
    return _enforce_plan(plan, stage="memory_write", context=context)


def enforce_output_plan(plan: GovernanceActionPlan, context: dict[str, Any]) -> RuntimeEnforcementResult:
    return _enforce_plan(plan, stage="output_commit", context=context)


def _attach_incident_if_needed(
    enforcement: RuntimeEnforcementResult,
    context_name: str,
    incident_id: str,
) -> RuntimeEnforcementResult:
    plan = enforcement.governance_plan
    if enforcement.blocked or enforcement.review_required or plan.open_incident:
        # context includes related signal ids in incident deterministically
        integration_result = enforcement.metadata.get("integration_result")
        if integration_result is not None:
            enforcement.incident = build_governance_incident(
                plan=plan,
                result=integration_result,
                context_name=context_name,
                incident_id=incident_id,
            )
    return enforcement


def guarded_pre_service_execution(
    payload: str,
    context: dict[str, Any] | None = None,
    runtime_commands: list[str] | None = None,
    memory_records: list[str] | None = None,
    face_baseline: dict[str, float] | None = None,
    anchor_integrity: float = 1.0,
) -> RuntimeEnforcementResult:
    """Guard pre-service flow: hook -> governance plan -> enforcement result."""

    runtime_context = context or {}
    guard_payload: dict[str, object] = {"input_text": payload}
    if runtime_commands:
        guard_payload["runtime_commands"] = runtime_commands
    if memory_records:
        guard_payload["memory_records"] = memory_records

    integration = pre_service_run_guard(
        payload=guard_payload,
        source_ref=str(runtime_context.get("source_ref", "pre_service")),
        base_face_scores=face_baseline,
        anchor_integrity=anchor_integrity,
    )
    plan = plan_pre_service_governance(integration)
    enforcement = enforce_pre_service_plan(plan, runtime_context)
    enforcement.metadata["integration_result"] = integration
    return _attach_incident_if_needed(enforcement, "pre_service", "enf-pre")


def guarded_post_service_execution(
    result_text: str,
    context: dict[str, Any] | None = None,
    emitted_commands: list[str] | None = None,
    face_baseline: dict[str, float] | None = None,
    anchor_integrity: float = 1.0,
) -> RuntimeEnforcementResult:
    """Guard post-service flow: hook -> governance plan -> enforcement result."""

    runtime_context = context or {}
    integration = post_service_run_guard(
        result_text=result_text,
        emitted_commands=emitted_commands,
        source_ref=str(runtime_context.get("source_ref", "post_service")),
        base_face_scores=face_baseline,
        anchor_integrity=anchor_integrity,
    )
    plan = plan_post_service_governance(integration)
    enforcement = enforce_post_service_plan(plan, runtime_context)
    enforcement.metadata["integration_result"] = integration
    return _attach_incident_if_needed(enforcement, "post_service", "enf-post")


def guarded_memory_write(
    records: list[str] | str,
    context: dict[str, Any] | None = None,
    face_baseline: dict[str, float] | None = None,
    anchor_integrity: float = 1.0,
) -> RuntimeEnforcementResult:
    """Guard memory-write flow: hook -> governance plan -> enforcement result."""

    runtime_context = context or {}
    integration = memory_write_guard(
        records=records,
        source_ref=str(runtime_context.get("source_ref", "memory_write")),
        base_face_scores=face_baseline,
        anchor_integrity=anchor_integrity,
    )
    plan = plan_memory_write_governance(integration)
    enforcement = enforce_memory_write_plan(plan, runtime_context)
    enforcement.metadata["integration_result"] = integration
    return _attach_incident_if_needed(enforcement, "memory_write", "enf-memory")


def guarded_output_commit(
    text: str,
    context: dict[str, Any] | None = None,
    face_baseline: dict[str, float] | None = None,
    anchor_integrity: float = 1.0,
) -> RuntimeEnforcementResult:
    """Guard output-commit flow: hook -> governance plan -> enforcement result."""

    runtime_context = context or {}
    integration = output_guard(
        text=text,
        source_ref=str(runtime_context.get("source_ref", "output_commit")),
        base_face_scores=face_baseline,
        anchor_integrity=anchor_integrity,
    )
    plan = plan_output_governance(integration)
    enforcement = enforce_output_plan(plan, runtime_context)
    enforcement.metadata["integration_result"] = integration
    return _attach_incident_if_needed(enforcement, "output_commit", "enf-output")
