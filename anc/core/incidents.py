"""Incident record model for continuity-preserving trust-plane forensics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from anc.core.models import GuardDecision, Posture


@dataclass(slots=True)
class IncidentRecord:
    """Immutable-style incident envelope for posture transitions and actions."""

    incident_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary: str = ""
    decision: GuardDecision | None = None
    related_signal_ids: list[str] = field(default_factory=list)
    before_posture: Posture = Posture.SAFE
    after_posture: Posture = Posture.SAFE
    metadata: dict[str, Any] = field(default_factory=dict)
