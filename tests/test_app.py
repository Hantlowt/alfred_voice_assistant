import sys
from pathlib import Path
from types import SimpleNamespace

from alfred_voice_assistant.app import (
    AlfredGate,
    RecentTranscripts,
    Terminal,
    deterministic_action,
    normalize_asr_transcript,
    run_microphone,
    safety_override_reason,
    transcript_for_intent,
)
from alfred_voice_assistant.home_assistant import SimulationExecutor


def test_recent_transcripts_normalizes_and_expires() -> None:
    recent = RecentTranscripts(ttl_seconds=3)

    assert recent.add_if_new("Allume la lumière !", now=10)
    assert not recent.add_if_new("  ALLUME la lumière... ", now=11)
    assert recent.add_if_new("Allume la lumière !", now=14)


def test_simulate_action_never_runs_a_real_action() -> None:
    executor = SimulationExecutor()

    assert executor.execute("allume les leds") == "🏠 [simulation uniquement] allume les leds"


def test_safety_override_catches_negation_and_reported_speech() -> None:
    assert safety_override_reason("N'allume surtout pas la lumière", "allume les lumières") == (
        "négation détectée"
    )
    assert safety_override_reason("Paul a dit d'allumer la lumière", "allume les lumières") == (
        "propos rapporté détecté"
    )
    assert safety_override_reason("J'aime quand la clim fonctionne", "allume la clim") == (
        "commande directe non détectée"
    )
    assert safety_override_reason(
        "Alfred, allumez les lumières s'il vous plaît", "allume les lumières"
    ) is None
    assert safety_override_reason("Pouvez-vous éteindre la clim", "éteins la clim") is None
    assert safety_override_reason(
        "Alfred éteint la Lorraine", "éteins l'halogène"
    ) == "équipement pris en charge non détecté clairement"


def test_explicit_commands_are_resolved_without_guessing_the_target() -> None:
    assert deterministic_action("alfred allume le salon") == "allume le salon"
    assert deterministic_action("Alfred, éteins le luminaire") == (
        "éteins le luminaire du salon"
    )
    assert deterministic_action("active les lumières du salon") == "allume le salon"
    assert deterministic_action("éteint la Lorraine") is None


def test_alfred_is_removed_before_laya_inference() -> None:
    assert transcript_for_intent("alfred allume le salon") == "allume le salon"
    assert transcript_for_intent("attends alfred allume les leds") == "allume les leds"


def test_alfred_gate_accepts_current_or_immediately_previous_phrase() -> None:
    gate = AlfredGate(required=True)

    assert not gate.evaluate("bonjour tout le monde").allowed
    assert gate.evaluate("Alfred").allowed
    previous = gate.evaluate("allume les lumières")
    assert previous.allowed
    assert "phrase précédente" in previous.status
    assert not gate.evaluate("allume les leds").allowed


def test_alfred_gate_can_be_disabled() -> None:
    gate = AlfredGate(required=False)

    assert gate.evaluate("allume les lumières").allowed
    assert gate.evaluate("éteins la clim").allowed


def test_known_halogen_asr_errors_are_normalized_before_intent_detection() -> None:
    assert normalize_asr_transcript("Alfred allume la le jeune") == "Alfred allume halogène"
    assert normalize_asr_transcript("éteins Eugène s'il te plaît") == (
        "éteins halogène s'il te plaît"
    )
    assert normalize_asr_transcript("Alfred allume le climat") == "Alfred allume la clim"
    assert normalize_asr_transcript("Alfred est un le salon") == "Alfred éteins le salon"
    assert normalize_asr_transcript("Alfred et une la clim") == "Alfred éteins la clim"
    assert normalize_asr_transcript("Alfred eteint les aides") == "Alfred éteins les leds"
    assert normalize_asr_transcript("attends alfred allume les l'aide") == (
        "attends alfred allume les leds"
    )
    assert normalize_asr_transcript("alfred étant la salle à manger") == (
        "alfred éteins la salle à manger"
    )
    assert normalize_asr_transcript("alfred est la logement") == "alfred éteins l'halogène"
    assert normalize_asr_transcript("alfred éteint la lorraine") == "alfred éteins l'halogène"
    assert normalize_asr_transcript("Alfred à l'homme le luminaire") == (
        "Alfred allume le luminaire"
    )
    assert normalize_asr_transcript("allume la clime") == "allume la clim"
    assert normalize_asr_transcript("c'est un très beau salon") == "c'est un très beau salon"
    assert normalize_asr_transcript("c'est un salon lumineux") == "c'est un salon lumineux"
    assert normalize_asr_transcript("allume l'halogène") == "allume l'halogène"


def test_asr_normalization_makes_etant_a_direct_command() -> None:
    transcript = normalize_asr_transcript("alfred étant la salle à manger")

    assert safety_override_reason(transcript, "éteins la salle à manger") is None


def test_command_after_alfred_ignores_leading_conversation() -> None:
    transcript = normalize_asr_transcript("attends alfred allume les l'aide")

    assert deterministic_action(transcript) == "allume les leds"
    assert safety_override_reason(transcript, "allume les leds") is None


def test_quiet_terminal_emits_nothing(capsys) -> None:
    terminal = Terminal(quiet=True)

    terminal.partial("texte partiel")
    terminal.line("résultat")
    terminal.clear_partial()

    assert capsys.readouterr().out == ""


def test_microphone_watchdog_reopens_stream_without_reloading_model(monkeypatch, capsys) -> None:
    state = {"model_loads": 0, "streams": 0}

    class FakeModel:
        def __init__(self, path: str) -> None:
            del path
            state["model_loads"] += 1

    class FakeRecognizer:
        def __init__(self, model: FakeModel, sample_rate: int) -> None:
            del model, sample_rate

    class FakeStream:
        def __init__(self, **kwargs) -> None:
            del kwargs
            state["streams"] += 1

        def __enter__(self):
            if state["streams"] == 2:
                raise KeyboardInterrupt
            return self

        def __exit__(self, *args) -> None:
            del args

    class FakeWorker:
        def __init__(self) -> None:
            self.started = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def close(self) -> None:
            self.closed = True

    fake_sounddevice = SimpleNamespace(
        PortAudioError=type("PortAudioError", (Exception,), {}),
        RawInputStream=FakeStream,
        query_devices=lambda device, kind: {
            "name": "micro test",
            "default_samplerate": 16_000,
        },
    )
    fake_vosk = SimpleNamespace(
        KaldiRecognizer=FakeRecognizer,
        Model=FakeModel,
        SetLogLevel=lambda level: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    monkeypatch.setitem(sys.modules, "vosk", fake_vosk)
    worker = FakeWorker()

    run_microphone(
        Path("modele-vosk"),
        worker,  # type: ignore[arg-type]
        Terminal(quiet=True),
        audio_watchdog_timeout=0.001,
        audio_retry_delay=0,
    )

    assert state == {"model_loads": 1, "streams": 2}
    assert worker.started
    assert worker.closed
    assert "Capture audio interrompue" in capsys.readouterr().err
