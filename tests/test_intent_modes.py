from alfred_voice_assistant.intent import ACTION_BY_PARTS, ACTIONS, LayaMlxIntentEngine


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
    engine._remove_do_nothing = False
    engine._agent = FakeAgent()

    result = engine.predict("éteins le luminaire du salon")

    assert result.intent == "éteins le luminaire du salon"
    assert result.confidence == 0.88
    assert [report.label for report in result.reports] == ["opération", "objet"]
    assert engine._agent.calls == [
        "éteins le luminaire du salon",
        "éteins le luminaire du salon",
    ]


class ChoiceAgent:
    def __init__(self) -> None:
        self.question: dict | None = None

    def predict(self, transcript: str, question: dict) -> dict:
        del transcript
        self.question = question
        choices = tuple(question["intent"]["criteria"])
        probabilities = dict.fromkeys(choices, 0.01)
        probabilities["allume les lumières"] = 0.87
        return {
            "answers": {
                "intent": {
                    "choice": "allume les lumières",
                    "probabilities": probabilities,
                }
            }
        }


def test_flat_mode_removes_do_nothing_from_the_laya_question() -> None:
    agent = ChoiceAgent()
    engine = object.__new__(LayaMlxIntentEngine)
    engine._mode = "flat"
    engine._remove_do_nothing = True
    engine._agent = agent
    engine._specialization = None
    engine._embed = None

    result = engine.predict("allume les lumières")

    assert agent.question is not None
    assert "do_nothing" not in agent.question["intent"]["criteria"]
    assert "do_nothing" not in result.reports[0].probabilities
    assert result.intent == "allume les lumières"


def test_every_supported_object_has_on_and_off_actions() -> None:
    expected = {
        "allume les lumières",
        "éteins les lumières",
        "allume l'halogène",
        "éteins l'halogène",
        "allume le salon",
        "éteins le salon",
        "allume la salle à manger",
        "éteins la salle à manger",
        "allume le luminaire du salon",
        "éteins le luminaire du salon",
        "allume les leds",
        "éteins les leds",
        "allume la clim",
        "éteins la clim",
    }

    assert set(ACTIONS) == expected | {"do_nothing"}
    assert set(ACTION_BY_PARTS.values()) == expected
