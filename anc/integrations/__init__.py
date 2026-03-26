"""ANC substrate-facing integration hooks for PhiKernel-style runtime guard points."""

from anc.integrations.phikernel_hooks import (
    memory_write_guard,
    output_guard,
    post_service_run_guard,
    pre_service_run_guard,
)
from anc.integrations.phikernel_enforcement import (
    RuntimeEnforcementResult,
    enforce_memory_write_plan,
    enforce_output_plan,
    enforce_post_service_plan,
    enforce_pre_service_plan,
    guarded_memory_write,
    guarded_output_commit,
    guarded_post_service_execution,
    guarded_pre_service_execution,
)
from anc.integrations.phikernel_governance import (
    GovernanceActionPlan,
    apply_governance_plan_to_context,
    build_governance_incident,
    plan_memory_write_governance,
    plan_output_governance,
    plan_post_service_governance,
    plan_pre_service_governance,
)
from anc.integrations.phikernel_soc_adapter import (
    DEFAULT_FACE_BASELINE,
    IntegrationGuardResult,
    build_incident_from_result,
    guard_input_text,
    guard_memory_records,
    guard_output_text,
    guard_runtime_commands,
)

__all__ = [
    "DEFAULT_FACE_BASELINE",
    "IntegrationGuardResult",
    "build_incident_from_result",
    "guard_input_text",
    "guard_memory_records",
    "guard_output_text",
    "guard_runtime_commands",
    "GovernanceActionPlan",
    "plan_pre_service_governance",
    "plan_post_service_governance",
    "plan_memory_write_governance",
    "plan_output_governance",
    "apply_governance_plan_to_context",
    "build_governance_incident",
    "RuntimeEnforcementResult",
    "enforce_pre_service_plan",
    "enforce_post_service_plan",
    "enforce_memory_write_plan",
    "enforce_output_plan",
    "guarded_pre_service_execution",
    "guarded_post_service_execution",
    "guarded_memory_write",
    "guarded_output_commit",
    "memory_write_guard",
    "output_guard",
    "post_service_run_guard",
    "pre_service_run_guard",
]
