"""ANC-specific continuity adapters and scorers for TIEKAT v57.7 primitives."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from anc.tiekat_v57 import (
    ContinuityDiagnostics,
    EpsilonSignature,
    MemoryCrystal,
    StateVector,
    clamp,
)


def _mean(values: Iterable[float]) -> float:
    seq = [float(v) for v in values]
    return sum(seq) / len(seq) if seq else 0.0


def validator_continuity_score(psi_window: Iterable[float], alpha_window: Iterable[float]) -> float:
    psi = _mean(psi_window)
    alpha = _mean(alpha_window)
    return clamp(0.6 * psi + 0.4 * alpha)


def network_continuity_score(lt_window: Iterable[float], permit_window: Iterable[int]) -> float:
    lt = _mean(lt_window)
    permit = _mean(float(v) for v in permit_window)
    return clamp(0.75 * lt + 0.25 * permit)


def branch_stability_score(path_scores: Iterable[float]) -> float:
    seq = [clamp(float(v)) for v in path_scores]
    if not seq:
        return 0.0
    spread = max(seq) - min(seq)
    return clamp(1.0 - spread)


def recursive_attractor_improvement(previous_score: float, current_score: float) -> float:
    prev = clamp(previous_score)
    cur = clamp(current_score)
    if prev <= 0.0:
        return cur
    delta = (cur - prev) / prev
    return clamp(0.5 + 0.5 * delta)


def build_epsilon_signature(
    epoch_start: int,
    epoch_end: int,
    regime_event: str,
    metrics_window: Iterable[Mapping[str, Any]],
) -> EpsilonSignature:
    rows = list(metrics_window)
    return EpsilonSignature(
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        validator_mean_psi=_mean(float(r.get("mean_psi_b", 0.0)) for r in rows),
        network_mean_lt=_mean(float(r.get("mean_lt", 0.0)) for r in rows),
        permit_rate=_mean(float(r.get("permit", 0.0)) for r in rows),
        regime_event=regime_event,
    )


def build_state_vector(
    validator_continuity: float,
    network_continuity: float,
    branch_stability: float,
    attractor_gain: float,
) -> StateVector:
    risk_load = clamp(1.0 - 0.5 * (validator_continuity + network_continuity))
    return StateVector(
        validator_continuity=clamp(validator_continuity),
        network_continuity=clamp(network_continuity),
        branch_stability=clamp(branch_stability),
        attractor_gain=clamp(attractor_gain),
        risk_load=risk_load,
    )


def build_diagnostics(state: StateVector, regime_event: str, weak_threshold: float = 0.58) -> ContinuityDiagnostics:
    continuity_score = clamp(0.5 * state.validator_continuity + 0.5 * state.network_continuity)
    return ContinuityDiagnostics(
        validator_continuity=state.validator_continuity,
        network_continuity=state.network_continuity,
        branch_stability=state.branch_stability,
        attractor_gain=state.attractor_gain,
        continuity_score=continuity_score,
        weak_continuity=continuity_score < weak_threshold,
        regime_event=regime_event,
    )


def build_memory_crystal(
    run_id: str,
    epoch_index: int,
    signature: EpsilonSignature,
    state: StateVector,
    diagnostics: ContinuityDiagnostics,
) -> MemoryCrystal:
    crystal_id = f"{run_id}:{epoch_index}:{signature.digest()[:16]}"
    return MemoryCrystal(
        crystal_id=crystal_id,
        run_id=run_id,
        epoch_index=epoch_index,
        epsilon_signature=signature,
        state_vector=state,
        diagnostics=diagnostics,
    )
