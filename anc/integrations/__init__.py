"""ANC substrate-facing integration hooks for PhiKernel-style runtime guard points."""

from anc.integrations.phikernel_hooks import (
    memory_write_guard,
    output_guard,
    post_service_run_guard,
    pre_service_run_guard,
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
    "memory_write_guard",
    "output_guard",
    "post_service_run_guard",
    "pre_service_run_guard",
]
