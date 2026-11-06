"""ANC -> proposed PhiShield R1 contract.

This is a local, advisory-only interoperability contract. It does not
authenticate a caller, mint capabilities, sign evidence, or execute policy.
A future PhiShield integration MUST use an authenticated trusted boundary
and its own independent authority/consent checks before any side effect.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from anc.core.models import Action, FACE_LABELS, Posture
from anc.integrations.phikernel_soc_adapter import IntegrationGuardResult

SCHEMA = "phi369.anc.phishield.advisory.v1"
MAX_TTL_SECONDS = 120
MAX_CLOCK_SKEW_SECONDS = 5
SCOPES = frozenset({"pre_service", "post_service", "memory_write", "output_commit"})
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
FIELDS = frozenset({
    "schema", "request_id", "scope", "posture", "recommended_action",
    "signal_count", "weakest_face", "weakest_face_score",
    "anchor_integrity", "observed_at", "not_after", "authority_granted",
})


@dataclass(frozen=True, slots=True)
class AncRiskAdvisoryV1:
    """Minimal, non-sensitive evidence for a particular request and action scope."""

    schema: str
    request_id: str
    scope: str
    posture: str
    recommended_action: str
    signal_count: int
    weakest_face: str
    weakest_face_score: float
    anchor_integrity: float
    observed_at: str
    not_after: str
    authority_granted: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Plain JSON-compatible fields, never a credential or execution grant."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReferencePolicyResult:
    """Example consumer disposition, NEVER an executable authorization."""

    disposition: str  # deny | review | candidate
    reason: str
    authoritative: bool = False


def _datetime_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("time must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamps must be strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid timestamp") from error
    return _datetime_utc(parsed)


def _validate_binding(request_id: str, scope: str) -> None:
    if not isinstance(request_id, str) or not REQUEST_ID_PATTERN.fullmatch(request_id):
        raise ValueError("invalid request_id")
    if not isinstance(scope, str) or scope not in SCOPES:
        raise ValueError("invalid scope")


def make_anc_advisory(
    guard_result: IntegrationGuardResult,
    *,
    request_id: str,
    scope: str,
    now: datetime | None = None,
    ttl_seconds: int = 30,
) -> AncRiskAdvisoryV1:
    """Adapt an existing ANC detector/scoring result into request-bound evidence."""
    _validate_binding(request_id, scope)
    if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= MAX_TTL_SECONDS:
        raise ValueError("invalid TTL")
    instant = _datetime_utc(now if now is not None else datetime.now(timezone.utc))
    state = guard_result.state
    advisory = AncRiskAdvisoryV1(
        schema=SCHEMA,
        request_id=request_id,
        scope=scope,
        posture=state.posture.value,
        recommended_action=guard_result.decision.action.value,
        signal_count=len(guard_result.signals),
        weakest_face=state.weakest_face,
        weakest_face_score=float(state.face_scores[state.weakest_face]),
        anchor_integrity=float(state.anchor_integrity),
        observed_at=instant.isoformat(),
        not_after=(instant + timedelta(seconds=ttl_seconds)).isoformat(),
    )
    # Make even internally-produced envelopes pass the same validation rules.
    parse_anc_advisory(advisory.to_payload(), request_id=request_id, scope=scope, now=instant)
    return advisory


def parse_anc_advisory(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    scope: str,
    now: datetime | None = None,
) -> AncRiskAdvisoryV1:
    """Strictly validate a versioned envelope. Validation is NOT authentication."""
    _validate_binding(request_id, scope)
    if not isinstance(payload, Mapping) or set(payload) != FIELDS:
        raise ValueError("advisory fields do not match R1 schema")
    if payload["schema"] != SCHEMA:
        raise ValueError("unsupported ANC advisory schema")
    if type(payload["authority_granted"]) is not bool or payload["authority_granted"]:
        raise ValueError("an advisory cannot grant authority")
    if payload["request_id"] != request_id or payload["scope"] != scope:
        raise ValueError("advisory request/scope mismatch")
    if payload["posture"] not in {p.value for p in Posture}:
        raise ValueError("invalid posture")
    if payload["recommended_action"] not in {a.value for a in Action}:
        raise ValueError("invalid action")
    if payload["weakest_face"] not in FACE_LABELS:
        raise ValueError("invalid face")
    if type(payload["signal_count"]) is not int or not 0 <= payload["signal_count"] <= 1000000:
        raise ValueError("invalid signal count")
    for field_name in ("weakest_face_score", "anchor_integrity"):
        value = payload[field_name]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid {field_name}")

    observed = _parse_time(payload["observed_at"])
    expiry = _parse_time(payload["not_after"])
    instant = _datetime_utc(now if now is not None else datetime.now(timezone.utc))
    lifetime = (expiry - observed).total_seconds()
    if not 0 < lifetime <= MAX_TTL_SECONDS:
        raise ValueError("invalid advisory validity interval")
    if observed > instant + timedelta(seconds=MAX_CLOCK_SKEW_SECONDS):
        raise ValueError("future-dated advisory")
    if instant >= expiry:
        raise ValueError("expired advisory")
    return AncRiskAdvisoryV1(**dict(payload))


def reference_phishield_gate(
    payload: Mapping[str, Any],
    *,
    request_id: str,
    scope: str,
    authority_verified: bool = False,
    consent_verified: bool = False,
    now: datetime | None = None,
) -> ReferencePolicyResult:
    """Inert PhiShield-like consumer for tests; no permission or side effects.

    authority_verified/consent_verified are test fixtures only. They MUST be
    supplied by a separately trusted identity/capability policy engine, NEVER
    from a client payload, an ANC result, or a coherence score.
    """
    try:
        advisory = parse_anc_advisory(payload, request_id=request_id, scope=scope, now=now)
    except (ValueError, TypeError, KeyError):
        return ReferencePolicyResult("deny", "missing, malformed, stale, or unbound advisory")
    if authority_verified is not True:
        return ReferencePolicyResult("deny", "independent authority not verified")
    if consent_verified is not True:
        return ReferencePolicyResult("deny", "explicit consent not verified")
    if advisory.posture in {Posture.HOSTILE.value, Posture.QUARANTINED.value}:
        return ReferencePolicyResult("deny", "hostile or quarantined risk posture")
    if advisory.recommended_action in {
        Action.REFUSE.value, Action.QUARANTINE.value, Action.SEAL.value,
    }:
        return ReferencePolicyResult("deny", "ANC recommends blocking")
    if (
        advisory.posture != Posture.SAFE.value
        or advisory.recommended_action != Action.ALLOW.value
        or advisory.signal_count > 0
    ):
        return ReferencePolicyResult("review", "risk evidence needs operator review")
    return ReferencePolicyResult("candidate", "no ANC risk flags; external authority still required")
