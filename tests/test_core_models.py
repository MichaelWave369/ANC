from datetime import datetime, timezone

import pytest

from anc.core.models import FACE_LABELS, Severity, SignalCategory, SignalLayer, ThreatSignal


BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_threat_signal_accepts_valid_fields() -> None:
    signal = ThreatSignal(
        signal_id="sig-1",
        layer=SignalLayer.INPUT,
        category=SignalCategory.INJECTION,
        severity=Severity.MEDIUM,
        confidence=0.8,
        affected_faces=[FACE_LABELS[0], FACE_LABELS[1]],
        evidence_excerpt="ignore policy",
        source_ref="prompt:42",
        timestamp=BASE_TS,
    )

    assert signal.confidence == 0.8
    assert signal.affected_faces == [FACE_LABELS[0], FACE_LABELS[1]]


def test_threat_signal_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        ThreatSignal(
            signal_id="sig-2",
            layer=SignalLayer.RUNTIME,
            category=SignalCategory.ANOMALY,
            severity=Severity.LOW,
            confidence=1.2,
        )


def test_threat_signal_rejects_unknown_face() -> None:
    with pytest.raises(ValueError):
        ThreatSignal(
            signal_id="sig-3",
            layer=SignalLayer.MEMORY,
            category=SignalCategory.CONTAMINATION,
            severity=Severity.HIGH,
            confidence=0.7,
            affected_faces=["unknown_face"],
        )
