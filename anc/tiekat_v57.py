"""TIEKAT v57.7 continuity primitives adapted for ANC v0.3.

This module intentionally focuses on deterministic, dependency-light primitives
that can be reused by the ANC simulator without importing any standalone CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

EPS = 1e-9


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to inclusive [lo, hi] bounds."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def stable_hash(payload: Mapping[str, Any] | Sequence[Any] | str) -> str:
    """Stable SHA-256 hash for structured deterministic payloads."""
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EpsilonSignature:
    """Compact epoch signature for continuity windows in validator simulations."""

    epoch_start: int
    epoch_end: int
    validator_mean_psi: float
    network_mean_lt: float
    permit_rate: float
    regime_event: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "epoch_start": int(self.epoch_start),
            "epoch_end": int(self.epoch_end),
            "validator_mean_psi": round(clamp(self.validator_mean_psi), 8),
            "network_mean_lt": round(clamp(self.network_mean_lt), 8),
            "permit_rate": round(clamp(self.permit_rate), 8),
            "regime_event": str(self.regime_event),
        }

    def digest(self) -> str:
        return stable_hash(self.to_payload())


@dataclass(frozen=True)
class StateVector:
    """Normalized state summary for one simulation window."""

    validator_continuity: float
    network_continuity: float
    branch_stability: float
    attractor_gain: float
    risk_load: float

    def to_list(self) -> List[float]:
        return [
            clamp(self.validator_continuity),
            clamp(self.network_continuity),
            clamp(self.branch_stability),
            clamp(self.attractor_gain),
            clamp(self.risk_load),
        ]

    def mean_strength(self) -> float:
        vals = self.to_list()
        return sum(vals) / len(vals)


@dataclass(frozen=True)
class ContinuityDiagnostics:
    """Diagnostics for continuity quality in a simulation path window."""

    validator_continuity: float
    network_continuity: float
    branch_stability: float
    attractor_gain: float
    continuity_score: float
    weak_continuity: bool
    regime_event: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "validator_continuity": round(clamp(self.validator_continuity), 8),
            "network_continuity": round(clamp(self.network_continuity), 8),
            "branch_stability": round(clamp(self.branch_stability), 8),
            "attractor_gain": round(clamp(self.attractor_gain), 8),
            "continuity_score": round(clamp(self.continuity_score), 8),
            "weak_continuity": bool(self.weak_continuity),
            "regime_event": self.regime_event,
        }


@dataclass(frozen=True)
class MemoryCrystal:
    """Persistent continuity checkpoint for recursive simulation learning."""

    crystal_id: str
    run_id: str
    epoch_index: int
    epsilon_signature: EpsilonSignature
    state_vector: StateVector
    diagnostics: ContinuityDiagnostics

    def to_payload(self) -> Dict[str, Any]:
        return {
            "crystal_id": self.crystal_id,
            "run_id": self.run_id,
            "epoch_index": int(self.epoch_index),
            "epsilon_signature": self.epsilon_signature.to_payload(),
            "state_vector": {
                "validator_continuity": self.state_vector.validator_continuity,
                "network_continuity": self.state_vector.network_continuity,
                "branch_stability": self.state_vector.branch_stability,
                "attractor_gain": self.state_vector.attractor_gain,
                "risk_load": self.state_vector.risk_load,
            },
            "diagnostics": self.diagnostics.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MemoryCrystal":
        sig = payload["epsilon_signature"]
        sv = payload["state_vector"]
        dg = payload["diagnostics"]
        return cls(
            crystal_id=str(payload["crystal_id"]),
            run_id=str(payload["run_id"]),
            epoch_index=int(payload["epoch_index"]),
            epsilon_signature=EpsilonSignature(
                epoch_start=int(sig["epoch_start"]),
                epoch_end=int(sig["epoch_end"]),
                validator_mean_psi=float(sig["validator_mean_psi"]),
                network_mean_lt=float(sig["network_mean_lt"]),
                permit_rate=float(sig["permit_rate"]),
                regime_event=str(sig["regime_event"]),
            ),
            state_vector=StateVector(
                validator_continuity=float(sv["validator_continuity"]),
                network_continuity=float(sv["network_continuity"]),
                branch_stability=float(sv["branch_stability"]),
                attractor_gain=float(sv["attractor_gain"]),
                risk_load=float(sv["risk_load"]),
            ),
            diagnostics=ContinuityDiagnostics(
                validator_continuity=float(dg["validator_continuity"]),
                network_continuity=float(dg["network_continuity"]),
                branch_stability=float(dg["branch_stability"]),
                attractor_gain=float(dg["attractor_gain"]),
                continuity_score=float(dg["continuity_score"]),
                weak_continuity=bool(dg["weak_continuity"]),
                regime_event=str(dg["regime_event"]),
            ),
        )


@dataclass
class ContinuumVault:
    """Persistence container for continuity memory crystals."""

    path: Path
    crystals: List[MemoryCrystal] = field(default_factory=list)

    def append(self, crystal: MemoryCrystal) -> None:
        self.crystals.append(crystal)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [c.to_payload() for c in self.crystals]
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self) -> None:
        if not self.path.exists():
            self.crystals = []
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.crystals = [MemoryCrystal.from_payload(item) for item in raw]

    def latest(self) -> MemoryCrystal | None:
        return self.crystals[-1] if self.crystals else None


def score_candidate_regime_path(diagnostics: ContinuityDiagnostics, penalty: float = 0.15) -> float:
    """Score one candidate regime path using weighted continuity factors."""
    base = (
        0.35 * clamp(diagnostics.validator_continuity)
        + 0.35 * clamp(diagnostics.network_continuity)
        + 0.20 * clamp(diagnostics.branch_stability)
        + 0.10 * clamp(diagnostics.attractor_gain)
    )
    weak_penalty = penalty if diagnostics.weak_continuity else 0.0
    return clamp(base - weak_penalty)


def compare_simulation_paths(candidates: Iterable[tuple[str, ContinuityDiagnostics]]) -> List[tuple[str, float]]:
    """Return candidate simulation paths ranked by deterministic score."""
    scored = [(name, score_candidate_regime_path(diag)) for name, diag in candidates]
    return sorted(scored, key=lambda item: (-item[1], item[0]))
