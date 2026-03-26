from anc.core.decisions import recommend_action
from anc.core.models import Action, CoherenceSecurityState, FACE_LABELS, Posture


def _state(posture: Posture, contamination_load: float = 0.0, recovery_vertex_score: float = 0.8) -> CoherenceSecurityState:
    return CoherenceSecurityState(
        posture=posture,
        face_scores={face: 0.8 for face in FACE_LABELS},
        weakest_face="foundation",
        field_average=0.8,
        field_variance=0.01,
        balanced=True,
        edge_flow_score=0.8,
        contamination_load=contamination_load,
        anchor_integrity=0.85,
        recovery_vertex_score=recovery_vertex_score,
        active_signal_count=1,
        recommended_action=Action.ALLOW,
    )


def test_decision_safe_allow() -> None:
    assert recommend_action(_state(Posture.SAFE)).action == Action.ALLOW


def test_decision_watch_warn() -> None:
    assert recommend_action(_state(Posture.WATCH)).action == Action.WARN


def test_decision_degraded_shadow_or_sandbox() -> None:
    assert recommend_action(_state(Posture.DEGRADED, recovery_vertex_score=0.6)).action == Action.SHADOW
    assert recommend_action(_state(Posture.DEGRADED, recovery_vertex_score=0.2)).action == Action.SANDBOX


def test_decision_hostile_quarantine_or_refuse() -> None:
    assert recommend_action(_state(Posture.HOSTILE, contamination_load=0.75)).action == Action.QUARANTINE
    assert recommend_action(_state(Posture.HOSTILE, contamination_load=0.4)).action == Action.REFUSE


def test_decision_quarantined_seal() -> None:
    decision = recommend_action(_state(Posture.QUARANTINED, contamination_load=0.9, recovery_vertex_score=0.1))
    assert decision.action == Action.SEAL
    assert decision.requires_recovery is True
