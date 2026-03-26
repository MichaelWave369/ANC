from anc.core.models import FACE_LABELS, Severity, SignalCategory, SignalLayer, ThreatSignal
from anc.core.scoring import compute_security_state


def _uniform_faces(score: float) -> dict[str, float]:
    return {face: score for face in FACE_LABELS}


def test_healthy_balanced_field_scores_safe() -> None:
    state = compute_security_state(base_face_scores=_uniform_faces(0.92), signals=[], anchor_integrity=0.95)

    assert state.posture.value == "safe"
    assert state.recommended_action.value == "allow"
    assert state.balanced is True
    assert state.contamination_load == 0.0


def test_one_face_collapse_degrades_field() -> None:
    faces = _uniform_faces(0.85)
    faces["memory"] = 0.18

    state = compute_security_state(base_face_scores=faces, signals=[], anchor_integrity=0.9)

    assert state.weakest_face == "memory"
    assert state.posture.value in {"watch", "degraded", "hostile"}


def test_fragmented_field_is_degraded_or_worse() -> None:
    faces = {
        "foundation": 0.95,
        "continuity": 0.45,
        "integrity": 0.90,
        "logic": 0.40,
        "memory": 0.88,
        "recovery": 0.35,
        "stability": 0.92,
        "observer": 0.42,
        "face_09": 0.85,
        "face_10": 0.33,
        "face_11": 0.80,
        "face_12": 0.30,
    }

    state = compute_security_state(base_face_scores=faces, signals=[], anchor_integrity=0.85)

    assert state.field_variance > 0.02
    assert state.posture.value in {"degraded", "hostile", "quarantined"}


def test_hostile_contamination_pushes_hostile_or_quarantined() -> None:
    signals = [
        ThreatSignal(
            signal_id="contam-1",
            layer=SignalLayer.MEMORY,
            category=SignalCategory.CONTAMINATION,
            severity=Severity.CRITICAL,
            confidence=0.95,
            affected_faces=["memory", "continuity", "integrity", "observer", "recovery", "stability"],
        ),
        ThreatSignal(
            signal_id="inject-1",
            layer=SignalLayer.INPUT,
            category=SignalCategory.INJECTION,
            severity=Severity.HIGH,
            confidence=0.95,
            affected_faces=["logic", "foundation", "face_10", "face_11", "face_12"],
        ),
    ]

    state = compute_security_state(base_face_scores=_uniform_faces(0.72), signals=signals, anchor_integrity=0.3)

    assert state.contamination_load > 0.2
    assert state.posture.value in {"hostile", "quarantined"}


def test_recovery_emergent_field_not_immediately_quarantined() -> None:
    faces = {
        "foundation": 0.62,
        "continuity": 0.58,
        "integrity": 0.66,
        "logic": 0.60,
        "memory": 0.54,
        "recovery": 0.64,
        "stability": 0.61,
        "observer": 0.59,
        "face_09": 0.63,
        "face_10": 0.57,
        "face_11": 0.62,
        "face_12": 0.56,
    }

    state = compute_security_state(base_face_scores=faces, signals=[], anchor_integrity=0.9)

    assert state.recovery_vertex_score > 0.45
    assert state.posture.value in {"watch", "degraded"}
