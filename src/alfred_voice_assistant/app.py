"""Capture Vosk streaming et traitement asynchrone des phrases finalisées."""

from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .home_assistant import ActionExecutor, SimulationExecutor
from .intent import IntentEngine

_SPACE_OR_PUNCTUATION = re.compile(r"[^\wÀ-ÿ]+", re.UNICODE)
_NEGATION = re.compile(
    r"\b(?:ne|n)\b(?:\s+\w+){0,8}\s+\b(?:pas|plus|jamais|rien)\b|"
    r"\b(?:pas|jamais)\s+(?:de|d|la|le|les|du|des)?\s*"
    r"(?:lumières?|halogène|salon|salle|leds?|clim(?:atisation)?)\b",
    re.IGNORECASE,
)
_REPORTED_SPEECH = re.compile(
    r"\b(?:dit|disait|demandé|demandait|ordonné|raconté)\b(?:\s+\w+){0,8}\s+\b(?:de|que|d)\b",
    re.IGNORECASE,
)
_POLITE_REQUEST = re.compile(
    r"^(?:s il (?:te|vous) plaît\s+)?(?:peux tu|pouvez vous|pourrais tu|pourriez vous|"
    r"est ce que tu peux|est ce que vous pouvez|merci de|veuillez|je veux que tu|"
    r"je voudrais que tu)\b",
    re.IGNORECASE,
)
_DIRECT_REQUEST = re.compile(
    r"^(?:s il (?:te|vous) plaît\s+)?(?:allum\w*|étein\w*|etein\w*|activ\w*|"
    r"désactiv\w*|desactiv\w*|coup\w*)\b",
    re.IGNORECASE,
)
_ACTION_TERM = re.compile(
    r"\b(?:allum\w*|étein\w*|etein\w*|activ\w*|désactiv\w*|desactiv\w*|coup\w*)\b",
    re.IGNORECASE,
)
_ALFRED = re.compile(r"\balfred\b", re.IGNORECASE)
_HALOGENE_ASR_ALIASES = (
    re.compile(r"\bla\s+le\s+jeune\b", re.IGNORECASE),
    re.compile(r"\beug[eèé]ne\b", re.IGNORECASE),
)
_CLIM_ASR_ALIAS_WITH_ARTICLE = re.compile(r"\ble\s+climat\b", re.IGNORECASE)
_CLIM_ASR_ALIAS = re.compile(r"\bclimat\b", re.IGNORECASE)
_ETEINS_ASR_ALIAS = re.compile(r"\best\s+un\b", re.IGNORECASE)


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


def normalize_asr_transcript(transcript: str) -> str:
    """Corrige quelques confusions Vosk connues avant la classification d'intention."""
    normalized = transcript
    for alias in _HALOGENE_ASR_ALIASES:
        normalized = alias.sub("halogène", normalized)
    normalized = _CLIM_ASR_ALIAS_WITH_ARTICLE.sub("la clim", normalized)
    normalized = _CLIM_ASR_ALIAS.sub("clim", normalized)
    normalized = _ETEINS_ASR_ALIAS.sub("éteins", normalized)
    return " ".join(normalized.split())


@dataclass(frozen=True, slots=True)
class AlfredAuthorization:
    allowed: bool
    status: str


class AlfredGate:
    """Autorise la phrase courante si « alfred » est présent ici ou juste avant."""

    def __init__(self, required: bool) -> None:
        self.required = required
        self._previous_had_keyword = False

    def evaluate(self, transcript: str) -> AlfredAuthorization:
        current_had_keyword = bool(_ALFRED.search(transcript))
        if not self.required:
            authorization = AlfredAuthorization(True, "mode désactivé — mot-clé non requis")
        elif current_had_keyword:
            authorization = AlfredAuthorization(True, "« alfred » détecté dans cette phrase")
        elif self._previous_had_keyword:
            authorization = AlfredAuthorization(
                True, "« alfred » détecté dans la phrase précédente"
            )
        else:
            authorization = AlfredAuthorization(False, "« alfred » absent")
        self._previous_had_keyword = current_had_keyword
        return authorization


def safety_override_reason(transcript: str, intent: str = "do_nothing") -> str | None:
    """Bloque les formes françaises dangereuses que le modèle générique confond parfois."""
    normalized = _SPACE_OR_PUNCTUATION.sub(" ", transcript.casefold()).strip()
    if _NEGATION.search(normalized):
        return "négation détectée"
    if _REPORTED_SPEECH.search(normalized):
        return "propos rapporté détecté"
    without_keyword = _ALFRED.sub(" ", normalized).strip()
    if intent != "do_nothing" and not (
        _DIRECT_REQUEST.search(without_keyword)
        or (_POLITE_REQUEST.search(without_keyword) and _ACTION_TERM.search(without_keyword))
    ):
        return "commande directe non détectée"
    return None


@dataclass(frozen=True, slots=True)
class PendingUtterance:
    transcript: str
    alfred: AlfredAuthorization


class IntentWorker:
    """Sérialise les appels MLX hors du thread de capture et sans bloquer le producteur."""

    def __init__(
        self,
        engine: IntentEngine,
        terminal: Terminal,
        confidence_threshold: float,
        duplicate_window: float,
        require_alfred: bool = False,
        action_executor: ActionExecutor | None = None,
        queue_size: int = 16,
    ) -> None:
        self._engine = engine
        self._terminal = terminal
        self._threshold = confidence_threshold
        self._recent = RecentTranscripts(duplicate_window)
        self._alfred_gate = AlfredGate(require_alfred)
        self._action_executor = action_executor or SimulationExecutor()
        self._queue: queue.Queue[PendingUtterance | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="laya-intents", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, transcript: str) -> None:
        normalized = normalize_asr_transcript(transcript)
        if normalized != transcript:
            self._terminal.line(f"🔧 Normalisation ASR         : {transcript!r} → {normalized!r}")
        alfred = self._alfred_gate.evaluate(normalized)
        if not self._recent.add_if_new(normalized):
            self._terminal.line("↪ Phrase dupliquée ignorée.")
            return
        item = PendingUtterance(normalized, alfred)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)
            self._terminal.line("⚠ File Laya saturée : la phrase la plus ancienne a été abandonnée.")

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._handle(item.transcript, item.alfred)
            except Exception as error:  # une erreur ne doit pas arrêter l'écoute
                self._terminal.line(f"Erreur de traitement : {type(error).__name__}: {error}")
            finally:
                self._queue.task_done()

    def _handle(
        self,
        transcript: str,
        alfred: AlfredAuthorization | None = None,
    ) -> None:
        alfred = alfred or AlfredAuthorization(True, "mode désactivé — mot-clé non requis")
        result = self._engine.predict(transcript)
        raw_confidence = result.confidence
        safety_reason = safety_override_reason(transcript, result.intent)
        model_accepted = (
            safety_reason is None
            and (result.intent == "do_nothing" or raw_confidence >= self._threshold)
        )
        candidate = result.intent if model_accepted else "do_nothing"
        blocked_by_alfred = candidate != "do_nothing" and not alfred.allowed
        final_action = "do_nothing" if blocked_by_alfred else candidate

        self._terminal.line(f"🧠 Action proposée par Laya : {result.intent}")
        self._terminal.line(f"🔑 Contrôle Alfred           : {alfred.status}")
        for report in result.reports:
            probabilities = " · ".join(
                f"{name}={probability:.1%}"
                for name, probability in sorted(
                    report.probabilities.items(), key=lambda item: item[1], reverse=True
                )
            )
            self._terminal.line(
                f"📊 Probabilités {report.label:<10} : {probabilities}"
            )
            if len(result.reports) > 1:
                self._terminal.line(
                    f"⏱  Latence {report.label:<15} : {report.latency_ms:.1f} ms"
                )
        self._terminal.line(f"⏱  Latence Laya totale       : {result.latency_ms:.1f} ms")

        reason = ""
        if safety_reason:
            reason = f"garde-fou : {safety_reason}"
        elif not model_accepted:
            reason = f"confiance sous le seuil de {self._threshold:.0%}"
        elif blocked_by_alfred:
            reason = "mot-clé « alfred » absent de cette phrase et de la précédente"
        elif final_action == "do_nothing":
            reason = "aucune action à exécuter"

        self._terminal.line(f"⚡ ACTION FINALE              : {final_action}")
        if final_action == "do_nothing":
            self._terminal.line(f"🚫 EXÉCUTION AUTORISÉE        : NON — {reason}")
        else:
            self._terminal.line("✅ EXÉCUTION AUTORISÉE        : OUI")
            self._terminal.line(self._action_executor.execute(final_action))


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
