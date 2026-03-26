"""Core topology-aware trust-plane model entities for ANC v1.0."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


FACE_LABELS: tuple[str, ...] = (
    "foundation",
    "continuity",
    "integrity",
    "logic",
    "memory",
    "recovery",
    "stability",
    "observer",
    "face_09",
    "face_10",
    "face_11",
    "face_12",
)


class SignalLayer(str, Enum):
    INPUT = "input"
    RUNTIME = "runtime"
    MEMORY = "memory"
    OUTPUT = "output"


class SignalCategory(str, Enum):
    JAILBREAK = "jailbreak"
    INJECTION = "injection"
    CONTAMINATION = "contamination"
    ANOMALY = "anomaly"
    EXFIL = "exfil"
    CAPTURE = "capture"
    OTHER = "other"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Posture(str, Enum):
    SAFE = "safe"
    WATCH = "watch"
    DEGRADED = "degraded"
    HOSTILE = "hostile"
    QUARANTINED = "quarantined"


class Action(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    SHADOW = "shadow"
    SANDBOX = "sandbox"
    QUARANTINE = "quarantine"
    REFUSE = "refuse"
    SEAL = "seal"


@dataclass(slots=True)
class ThreatSignal:
    """Normalized threat signal used for deterministic topology-aware scoring."""

    signal_id: str
    layer: SignalLayer
    category: SignalCategory
    severity: Severity
    confidence: float
    affected_faces: list[str] = field(default_factory=list)
    evidence_excerpt: str = ""
    source_ref: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must be in [0, 1]"
            raise ValueError(msg)
        invalid = [face for face in self.affected_faces if face not in FACE_LABELS]
        if invalid:
            msg = f"affected_faces contains unknown faces: {invalid}"
            raise ValueError(msg)


@dataclass(slots=True)
class CoherenceSecurityState:
    """Deterministic security state derived from 12-face topology metrics."""

    posture: Posture
    face_scores: dict[str, float]
    weakest_face: str
    field_average: float
    field_variance: float
    balanced: bool
    edge_flow_score: float
    contamination_load: float
    anchor_integrity: float
    recovery_vertex_score: float
    active_signal_count: int
    recommended_action: Action


@dataclass(slots=True)
class GuardDecision:
    """Policy action recommendation derived from current security posture."""

    action: Action
    rationale: str
    triggered_by: list[str]
    operator_visible: bool
    requires_recovery: bool


def normalized_face_map(face_scores: Mapping[str, float]) -> dict[str, float]:
    """Return 12-face map with values clamped to [0, 1]."""

    missing = [face for face in FACE_LABELS if face not in face_scores]
    extra = [face for face in face_scores if face not in FACE_LABELS]
    if missing or extra:
        msg = f"face_scores must contain exactly 12 known faces. missing={missing} extra={extra}"
        raise ValueError(msg)

    return {face: max(0.0, min(1.0, float(face_scores[face]))) for face in FACE_LABELS}
