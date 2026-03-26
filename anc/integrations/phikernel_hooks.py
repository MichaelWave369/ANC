"""PhiKernel SoC style guard hook wrappers over the ANC integration adapter."""

from __future__ import annotations

from anc.integrations.phikernel_soc_adapter import (
    DEFAULT_FACE_BASELINE,
    IntegrationGuardResult,
    guard_input_text,
    guard_memory_records,
    guard_output_text,
)


def pre_service_run_guard(
    payload: str | dict[str, object],
    source_ref: str = "",
    base_face_scores: dict[str, float] | None = None,
    anchor_integrity: float = 0.90,
) -> IntegrationGuardResult:
    """Guard pre-run service request payloads before execution."""

    if isinstance(payload, dict):
        input_text = str(payload.get("input_text", ""))
        memory_records = payload.get("memory_records")
        runtime_commands = payload.get("runtime_commands")
        return guard_input_text(
            text=input_text,
            source_ref=source_ref,
            memory_records=memory_records if isinstance(memory_records, list) else None,
            runtime_commands=runtime_commands if isinstance(runtime_commands, (list, str)) else None,
            base_face_scores=base_face_scores or DEFAULT_FACE_BASELINE,
            anchor_integrity=anchor_integrity,
        )

    return guard_input_text(
        text=payload,
        source_ref=source_ref,
        base_face_scores=base_face_scores or DEFAULT_FACE_BASELINE,
        anchor_integrity=anchor_integrity,
    )


def post_service_run_guard(
    result_text: str,
    emitted_commands: list[str] | str | None = None,
    source_ref: str = "",
    base_face_scores: dict[str, float] | None = None,
    anchor_integrity: float = 0.90,
) -> IntegrationGuardResult:
    """Guard service result/output surfaces after execution."""

    return guard_output_text(
        text=result_text,
        emitted_commands=emitted_commands,
        source_ref=source_ref,
        base_face_scores=base_face_scores or DEFAULT_FACE_BASELINE,
        anchor_integrity=anchor_integrity,
    )


def memory_write_guard(
    records: list[str] | str,
    source_ref: str = "",
    base_face_scores: dict[str, float] | None = None,
    anchor_integrity: float = 0.90,
) -> IntegrationGuardResult:
    """Guard memory writes/appends before commit."""

    return guard_memory_records(
        records=records,
        source_ref=source_ref,
        base_face_scores=base_face_scores or DEFAULT_FACE_BASELINE,
        anchor_integrity=anchor_integrity,
    )


def output_guard(
    text: str,
    source_ref: str = "",
    base_face_scores: dict[str, float] | None = None,
    anchor_integrity: float = 0.90,
) -> IntegrationGuardResult:
    """Guard output text before final display/commit."""

    return guard_output_text(
        text=text,
        source_ref=source_ref,
        base_face_scores=base_face_scores or DEFAULT_FACE_BASELINE,
        anchor_integrity=anchor_integrity,
    )
