from anc.core.models import Action, CoherenceSecurityState, FACE_LABELS, Posture
from anc.core.posture import determine_posture


def _state(**overrides) -> CoherenceSecurityState:
    faces = {face: 0.9 for face in FACE_LABELS}
    base = CoherenceSecurityState(
        posture=Posture.SAFE,
        face_scores=faces,
        weakest_face="foundation",
        field_average=0.9,
        field_variance=0.001,
        balanced=True,
        edge_flow_score=0.95,
        contamination_load=0.0,
        anchor_integrity=0.95,
        recovery_vertex_score=0.9,
        active_signal_count=0,
        recommended_action=Action.ALLOW,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_determine_posture_safe() -> None:
    assert determine_posture(_state()) == Posture.SAFE


def test_determine_posture_watch() -> None:
    assert determine_posture(_state(active_signal_count=1, field_average=0.84)) == Posture.WATCH


def test_determine_posture_degraded() -> None:
    assert determine_posture(_state(field_average=0.62, field_variance=0.05, balanced=False)) == Posture.DEGRADED


def test_determine_posture_hostile() -> None:
    faces = {face: 0.8 for face in FACE_LABELS}
    faces["logic"] = 0.2
    assert (
        determine_posture(
            _state(face_scores=faces, weakest_face="logic", field_average=0.52, contamination_load=0.66),
        )
        == Posture.HOSTILE
    )


def test_determine_posture_quarantined() -> None:
    faces = {face: 0.5 for face in FACE_LABELS}
    faces["memory"] = 0.05
    assert (
        determine_posture(
            _state(
                face_scores=faces,
                weakest_face="memory",
                field_average=0.24,
                contamination_load=0.9,
                recovery_vertex_score=0.1,
                anchor_integrity=0.2,
            ),
        )
        == Posture.QUARANTINED
    )
