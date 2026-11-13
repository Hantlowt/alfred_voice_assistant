import sys
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

from alfred_voice_assistant.app import (
    AlfredGate,
    KyutaiAudioEvent,
    KyutaiTranscriber,
    RecentTranscripts,
    Terminal,
    normalize_asr_transcript,
    run_microphone,
    transcript_for_intent,
)
from alfred_voice_assistant.home_assistant import SimulationExecutor


def test_recent_transcripts_normalizes_and_expires() -> None:
    recent = RecentTranscripts(ttl_seconds=3)

    assert recent.add_if_new("Allume la lumière !", now=10)
    assert not recent.add_if_new("  ALLUME la lumière... ", now=11)
    assert recent.add_if_new("Allume la lumière !", now=14)


def test_kyutai_stream_flushes_delayed_tokens_at_end_of_utterance() -> None:
    class FakeAudio:
        shape = (1, 1, 1920)

        def split(self, size: int, dim: int):
            assert size == 1920
            assert dim == 2
            return (self,)

    class FakeToken:
        def __init__(self, token_id: int) -> None:
            self.token_id = token_id

        def __getitem__(self, key):
            del key
            return self

        def item(self) -> int:
            return self.token_id

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def streaming(self, batch_size: int):
            assert batch_size == 1
            return nullcontext()

        def step(self, codes):
            del codes
            self.calls += 1
            token_id = {2: 10, 3: 11}.get(self.calls, 0)
            return FakeToken(token_id)

    generator = FakeGenerator()
    transcriber = object.__new__(KyutaiTranscriber)
    transcriber._torch = SimpleNamespace(
        no_grad=nullcontext,
        zeros=lambda *args, **kwargs: FakeAudio(),
        float32="float32",
        nn=SimpleNamespace(functional=SimpleNamespace(pad=lambda audio, padding: audio)),
    )
    transcriber._device = "cpu"
    transcriber._checkpoint = SimpleNamespace(stt_config={"audio_delay_seconds": 0.5})
    transcriber._mimi = SimpleNamespace(
        sample_rate=24_000,
        frame_rate=12.5,
        encode=lambda frame: frame,
        streaming=lambda batch_size: nullcontext(),
    )
    transcriber._tokenizer = SimpleNamespace(
        eos_id=lambda: 2,
        id_to_piece=lambda token_id: {10: "▁Alfred", 11: "▁allume"}[token_id],
    )
    transcriber._lm = object()
    transcriber._models = SimpleNamespace(LMGen=lambda *args, **kwargs: generator)
    transcriber._audio_tensor = lambda pcm, sample_rate: FakeAudio()
    events: Queue[KyutaiAudioEvent | None] = Queue()
    events.put(KyutaiAudioEvent(b"audio", 24_000, start_of_utterance=True))
    events.put(KyutaiAudioEvent(b"", 24_000, end_of_utterance=True))
    events.put(None)
    partials: list[str] = []
    finals: list[str] = []

    transcriber.run_stream(events, partials.append, finals.append)

    assert finals == ["Alfred allume"]
    assert "Alfred" in partials
    assert generator.calls == 10  # 2 appels initiaux + 8 frames de flush accéléré.


def test_simulate_action_never_runs_a_real_action() -> None:
    executor = SimulationExecutor()

    assert executor.execute("allume les leds") == "🏠 [simulation uniquement] allume les leds"


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
