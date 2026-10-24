from alfred_voice_assistant.app import AlfredAuthorization, IntentWorker
from alfred_voice_assistant.intent import ACTIONS, IntentResult, ProbabilityReport


class FakeTerminal:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        self.lines.append(text)


class FakeEngine:
    probability = 0.60

    def predict(self, transcript: str) -> IntentResult:
        del transcript
        probabilities = dict.fromkeys(ACTIONS, 0.01)
        probabilities["allume les lumières"] = self.probability
        probabilities["do_nothing"] = 0.27
        return IntentResult(
            intent="allume les lumières",
            confidence=self.probability,
            reports=(ProbabilityReport("actions", probabilities, 12.5),),
            latency_ms=12.5,
        )


def test_threshold_falls_back_to_do_nothing() -> None:
    terminal = FakeTerminal()
    worker = IntentWorker(FakeEngine(), terminal, confidence_threshold=0.7, duplicate_window=3)

    worker._handle("allume")

    assert any("ACTION FINALE" in line and "do_nothing" in line for line in terminal.lines)
    assert any("EXÉCUTION SIMULÉE" in line and "NON" in line for line in terminal.lines)


def test_required_alfred_blocks_an_otherwise_valid_action() -> None:
    terminal = FakeTerminal()
    engine = FakeEngine()
    engine.probability = 0.90
    worker = IntentWorker(engine, terminal, confidence_threshold=0.7, duplicate_window=3)

    worker._handle(
        "allume les lumières",
        AlfredAuthorization(False, "« alfred » absent"),
    )

    assert any("Action proposée" in line and "allume les lumières" in line for line in terminal.lines)
    assert any("ACTION FINALE" in line and "do_nothing" in line for line in terminal.lines)
    assert any("mot-clé" in line for line in terminal.lines)


def test_required_alfred_allows_the_simulated_action() -> None:
    terminal = FakeTerminal()
    engine = FakeEngine()
    engine.probability = 0.90
    worker = IntentWorker(engine, terminal, confidence_threshold=0.7, duplicate_window=3)

    worker._handle(
        "Alfred allume les lumières",
        AlfredAuthorization(True, "« alfred » détecté dans cette phrase"),
    )

    assert any("ACTION FINALE" in line and "allume les lumières" in line for line in terminal.lines)
    assert any("EXÉCUTION SIMULÉE" in line and "OUI" in line for line in terminal.lines)
    assert any("simulation uniquement" in line for line in terminal.lines)


def test_submit_normalizes_vosk_alias_before_queuing() -> None:
    terminal = FakeTerminal()
    worker = IntentWorker(FakeEngine(), terminal, confidence_threshold=0.7, duplicate_window=3)

    worker.submit("Alfred allume la le jeune")
    pending = worker._queue.get_nowait()

    assert pending is not None
    assert pending.transcript == "Alfred allume halogène"
    assert any("Normalisation ASR" in line for line in terminal.lines)
