from alfred_voice_assistant.intent import LayaMlxIntentEngine


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, transcript: str, question: dict) -> dict:
        self.calls.append(transcript)
        if "operation" in question:
            answer = {
                "choice": "éteindre",
                "probabilities": {"allumer": 0.02, "éteindre": 0.93, "do_nothing": 0.05},
            }
            return {"answers": {"operation": answer}}
        answer = {
            "choice": "le luminaire du salon",
            "probabilities": {
                "les lumières": 0.02,
                "l'halogène": 0.01,
                "le salon": 0.05,
                "la salle à manger": 0.01,
                "le luminaire du salon": 0.88,
                "les leds": 0.01,
                "la clim": 0.01,
                "do_nothing": 0.01,
            },
        }
        return {"answers": {"objet": answer}}


def test_factorized_mode_calls_laya_twice_and_combines_answers() -> None:
    engine = object.__new__(LayaMlxIntentEngine)
    engine._mode = "factorized"
    engine._agent = FakeAgent()

    result = engine.predict("éteins le luminaire du salon")

    assert result.intent == "éteins le luminaire du salon"
    assert result.confidence == 0.88
    assert [report.label for report in result.reports] == ["opération", "objet"]
    assert engine._agent.calls == [
        "éteins le luminaire du salon",
        "éteins le luminaire du salon",
    ]
