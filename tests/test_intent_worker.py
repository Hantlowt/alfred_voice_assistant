from alfred_voice_assistant.app import IntentWorker
from alfred_voice_assistant.intent import ACTIONS, IntentResult


class FakeTerminal:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        self.lines.append(text)


class FakeEngine:
    def predict(self, transcript: str) -> IntentResult:
        del transcript
        probabilities = dict.fromkeys(ACTIONS, 0.02)
        probabilities["turn_on_light"] = 0.60
        probabilities["do_nothing"] = 0.30
        return IntentResult("turn_on_light", probabilities, 12.5)


def test_threshold_falls_back_to_do_nothing() -> None:
    terminal = FakeTerminal()
    worker = IntentWorker(FakeEngine(), terminal, confidence_threshold=0.7, duplicate_window=3)

    worker._handle("allume")

    assert any("Intention" in line and "do_nothing" in line for line in terminal.lines)
    assert not any("Lumière allumée" in line for line in terminal.lines)
