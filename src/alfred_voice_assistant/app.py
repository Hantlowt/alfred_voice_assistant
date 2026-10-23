"""Capture Vosk streaming et traitement asynchrone des phrases finalisées."""

from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .intent import IntentEngine

_SPACE_OR_PUNCTUATION = re.compile(r"[^\wÀ-ÿ]+", re.UNICODE)
_NEGATION = re.compile(
    r"\b(?:ne|n)\b(?:\s+\w+){0,8}\s+\b(?:pas|plus|jamais|rien)\b|"
    r"\b(?:pas|jamais)\s+(?:de|d|la|le|les|du|des)?\s*(?:lumière|musique)\b",
    re.IGNORECASE,
)
_REPORTED_SPEECH = re.compile(
    r"\b(?:dit|disait|demandé|demandait|ordonné|raconté)\b(?:\s+\w+){0,8}\s+\b(?:de|que|d)\b",
    re.IGNORECASE,
)
_POLITE_REQUEST = re.compile(
    r"^(?:s il te plaît\s+)?(?:peux tu|pourrais tu|est ce que tu peux|merci de|veuillez|"
    r"je veux que tu|je voudrais que tu)\b",
    re.IGNORECASE,
)
_DIRECT_REQUESTS = {
    "turn_on_light": re.compile(r"^(?:s il te plaît\s+)?(?:allume|active|mets)\b", re.IGNORECASE),
    "turn_off_light": re.compile(r"^(?:s il te plaît\s+)?(?:éteins|coupe|désactive)\b", re.IGNORECASE),
    "get_weather": re.compile(
        r"^(?:s il te plaît\s+)?(?:météo|la météo|quel temps|quelle météo|fait il|"
        r"va t il (?:pleuvoir|neiger)|pleut il|donne moi (?:la )?météo|affiche (?:la )?météo)\b",
        re.IGNORECASE,
    ),
    "play_music": re.compile(
        r"^(?:s il te plaît\s+)?(?:joue|lance|mets|démarre|reprends|fais jouer)\b",
        re.IGNORECASE,
    ),
    "stop_music": re.compile(
        r"^(?:s il te plaît\s+)?(?:arrête|stoppe|coupe|pause|mets en pause)\b",
        re.IGNORECASE,
    ),
}
_ACTION_TERMS = {
    "turn_on_light": re.compile(r"\b(?:allum\w*|activ\w*)\b", re.IGNORECASE),
    "turn_off_light": re.compile(r"\b(?:étein\w*|coup\w*|désactiv\w*)\b", re.IGNORECASE),
    "get_weather": re.compile(r"\b(?:météo|temps|pleuv\w*|neig\w*)\b", re.IGNORECASE),
    "play_music": re.compile(r"\b(?:jou\w*|lanc\w*|musique|repren\w*)\b", re.IGNORECASE),
    "stop_music": re.compile(r"\b(?:arr[eê]t\w*|stopp\w*|pause|coup\w*)\b", re.IGNORECASE),
}


class Terminal:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._interactive = sys.stdout.isatty()
        self._partial = ""

    def partial(self, text: str) -> None:
        with self._lock:
            self._partial = text
            if self._interactive:
                print(f"\r\033[2K… {text}", end="", flush=True)
            elif text:
                print(f"[partiel] {text}", flush=True)

    def clear_partial(self) -> None:
        with self._lock:
            self._partial = ""
            if self._interactive:
                print("\r\033[2K", end="", flush=True)

    def line(self, text: str = "") -> None:
        with self._lock:
            if self._interactive:
                print("\r\033[2K", end="")
            print(text, flush=True)
            if self._interactive and self._partial:
                print(f"… {self._partial}", end="", flush=True)


class RecentTranscripts:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._seen: dict[str, float] = {}

    @staticmethod
    def normalize(text: str) -> str:
        return _SPACE_OR_PUNCTUATION.sub(" ", text.casefold()).strip()

    def add_if_new(self, text: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        normalized = self.normalize(text)
        self._seen = {key: at for key, at in self._seen.items() if current - at < self._ttl}
        if not normalized or normalized in self._seen:
            return False
        self._seen[normalized] = current
        return True


def safety_override_reason(transcript: str, intent: str = "do_nothing") -> str | None:
    """Bloque les formes françaises dangereuses que le modèle générique confond parfois."""
    normalized = _SPACE_OR_PUNCTUATION.sub(" ", transcript.casefold()).strip()
    if _NEGATION.search(normalized):
        return "négation détectée"
    if _REPORTED_SPEECH.search(normalized):
        return "propos rapporté détecté"
    direct_pattern = _DIRECT_REQUESTS.get(intent)
    action_terms = _ACTION_TERMS.get(intent)
    if direct_pattern and not (
        direct_pattern.search(normalized)
        or (_POLITE_REQUEST.search(normalized) and action_terms and action_terms.search(normalized))
    ):
        return "commande directe non détectée"
    return None


class IntentWorker:
    """Sérialise les appels MLX hors du thread de capture et sans bloquer le producteur."""

    def __init__(
        self,
        engine: IntentEngine,
        terminal: Terminal,
        confidence_threshold: float,
        duplicate_window: float,
        queue_size: int = 16,
    ) -> None:
        self._engine = engine
        self._terminal = terminal
        self._threshold = confidence_threshold
        self._recent = RecentTranscripts(duplicate_window)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="laya-intents", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, transcript: str) -> None:
        if not self._recent.add_if_new(transcript):
            self._terminal.line("↪ Phrase dupliquée ignorée.")
            return
        try:
            self._queue.put_nowait(transcript)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(transcript)
            self._terminal.line("⚠ File Laya saturée : la phrase la plus ancienne a été abandonnée.")

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            transcript = self._queue.get()
            try:
                if transcript is None:
                    return
                self._handle(transcript)
            except Exception as error:  # une erreur d'inférence ne doit pas arrêter l'écoute
                self._terminal.line(f"Erreur Laya : {type(error).__name__}: {error}")
            finally:
                self._queue.task_done()

    def _handle(self, transcript: str) -> None:
        result = self._engine.predict(transcript)
        raw_confidence = result.probabilities[result.intent]
        safety_reason = safety_override_reason(transcript, result.intent)
        accepted = (
            safety_reason is None
            and (result.intent == "do_nothing" or raw_confidence >= self._threshold)
        )
        intent = result.intent if accepted else "do_nothing"

        probabilities = " · ".join(
            f"{name}={probability:.1%}"
            for name, probability in sorted(
                result.probabilities.items(), key=lambda item: item[1], reverse=True
            )
        )
        suffix = ""
        if safety_reason:
            suffix = f" (Laya: {result.intent}, garde-fou: {safety_reason})"
        elif not accepted:
            suffix = f" (Laya: {result.intent}, sous le seuil {self._threshold:.0%})"
        self._terminal.line(f"🎯 Intention     : {intent}{suffix}")
        self._terminal.line(f"📊 Probabilités : {probabilities}")
        self._terminal.line(f"⏱  Latence Laya : {result.latency_ms:.1f} ms")
        simulate_action(intent, self._terminal.line)


def simulate_action(intent: str, output: Callable[[str], None] = print) -> None:
    messages = {
        "turn_on_light": "💡 [simulation] Lumière allumée.",
        "turn_off_light": "🌑 [simulation] Lumière éteinte.",
        "get_weather": "🌦  [simulation] Consultation de la météo.",
        "play_music": "▶️  [simulation] Lecture de la musique.",
        "stop_music": "⏹  [simulation] Arrêt de la musique.",
    }
    if intent in messages:
        output(messages[intent])


def run_microphone(
    vosk_model_path: Path,
    worker: IntentWorker,
    terminal: Terminal,
    device: int | str | None = None,
) -> None:
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model, SetLogLevel

    SetLogLevel(-1)
    device_info: Any = sd.query_devices(device, "input")
    sample_rate = int(device_info["default_samplerate"])
    recognizer = KaldiRecognizer(Model(str(vosk_model_path)), sample_rate)
    audio: queue.Queue[bytes] = queue.Queue(maxsize=64)
    dropped_blocks = 0

    def callback(indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        del frames, callback_time
        nonlocal dropped_blocks
        if status:
            dropped_blocks += 1
        try:
            audio.put_nowait(bytes(indata))
        except queue.Full:
            dropped_blocks += 1

    terminal.line(f"Microphone : {device_info['name']} · {sample_rate} Hz")
    terminal.line("Écoute continue — Ctrl+C pour quitter.")
    last_partial = ""
    worker.start()
    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=4000,
            device=device,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            while True:
                chunk = audio.get()
                if dropped_blocks:
                    terminal.line(f"⚠ {dropped_blocks} bloc(s) audio perdu(s).")
                    dropped_blocks = 0
                if recognizer.AcceptWaveform(chunk):
                    text = str(json.loads(recognizer.Result()).get("text", "")).strip()
                    terminal.clear_partial()
                    last_partial = ""
                    if text:
                        terminal.line(f"🗣  Transcription : {text}")
                        worker.submit(text)
                else:
                    partial = str(json.loads(recognizer.PartialResult()).get("partial", "")).strip()
                    if partial != last_partial:
                        terminal.partial(partial)
                        last_partial = partial
    except KeyboardInterrupt:
        terminal.clear_partial()
        terminal.line("Arrêt demandé.")
    finally:
        worker.close()
