"""Capture audio, transcription Kyutai/Vosk et traitement asynchrone des phrases."""

from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .home_assistant import ActionExecutor, SimulationExecutor
from .intent import ACTION_BY_PARTS, IntentEngine


@dataclass(frozen=True, slots=True)
class KyutaiAudioEvent:
    pcm: bytes
    sample_rate: int
    start_of_utterance: bool = False
    end_of_utterance: bool = False


class KyutaiTranscriber:
    """Modèle Kyutai chargé une fois et alimenté en continu par blocs audio."""

    def __init__(self, model_path: Path) -> None:
        import torch
        from moshi import models

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = models.loaders.CheckpointInfo.from_hf_repo(
            "kyutai/stt-1b-en_fr",
            config_path=model_path / "config.json",
            moshi_weights=model_path / "model.safetensors",
            mimi_weights=model_path / "mimi-pytorch-e351c8d8@125.safetensors",
            tokenizer=model_path / "tokenizer_en_fr_audio_8000.model",
        )
        self._checkpoint = checkpoint
        self._mimi = checkpoint.get_mimi(device=self._device)
        self._tokenizer = checkpoint.get_text_tokenizer()
        self._lm = checkpoint.get_moshi(device=self._device, dtype=torch.bfloat16)
        self._models = models
        self.sample_rate = int(self._mimi.sample_rate)

    @property
    def device(self) -> str:
        return self._device

    def _audio_tensor(self, pcm: bytes, sample_rate: int):
        torch = self._torch
        samples = np.frombuffer(pcm, dtype=np.int16).copy()
        audio = torch.from_numpy(samples).to(self._device, dtype=torch.float32)
        audio = (audio / 32768.0).view(1, 1, -1)
        if sample_rate != self.sample_rate:
            target_length = round(audio.shape[-1] * self.sample_rate / sample_rate)
            audio = torch.nn.functional.interpolate(
                audio,
                size=target_length,
                mode="linear",
                align_corners=False,
            )
        return audio

    def run_stream(
        self,
        events: queue.Queue[KyutaiAudioEvent | None],
        on_partial: Callable[[str], None],
        on_final: Callable[[str], None],
    ) -> None:
        """Consomme l'audio en temps réel et vide le délai du modèle à chaque fin de tour."""
        torch = self._torch
        frame_size = int(self._mimi.sample_rate / self._mimi.frame_rate)
        audio_delay = float(self._checkpoint.stt_config.get("audio_delay_seconds", 0.5))
        flush_frames = max(1, round((audio_delay + 0.1) * self._mimi.frame_rate))
        silence_frame = torch.zeros(
            (1, 1, frame_size),
            dtype=torch.float32,
            device=self._device,
        )

        generator = self._models.LMGen(
            self._lm,
            temp=0,
            temp_text=0,
            use_sampling=False,
        )
        pieces: list[str] = []
        first_frame = True

        def process_frame(frame) -> None:
            nonlocal first_frame
            codes = self._mimi.encode(frame)
            if first_frame:
                generator.step(codes)
                first_frame = False
            tokens = generator.step(codes)
            if tokens is None:
                return
            token_id = int(tokens[0, 0, 0].item())
            if token_id not in (0, 3, self._tokenizer.eos_id()):
                pieces.append(self._tokenizer.id_to_piece(token_id).replace("▁", " "))
                on_partial("".join(pieces).strip())

        with torch.no_grad(), self._mimi.streaming(1), generator.streaming(1):
            while True:
                event = events.get()
                try:
                    if event is None:
                        return
                    if event.start_of_utterance:
                        pieces.clear()
                        on_partial("")
                    if event.pcm:
                        audio = self._audio_tensor(event.pcm, event.sample_rate)
                        remainder = audio.shape[-1] % frame_size
                        if remainder:
                            audio = torch.nn.functional.pad(audio, (0, frame_size - remainder))
                        for frame in audio.split(frame_size, dim=2):
                            process_frame(frame)
                    if event.end_of_utterance:
                        # « Flush trick » Kyutai : les 500 ms de retard sont calculées
                        # immédiatement au lieu d'être attendues en temps réel.
                        for _ in range(flush_frames):
                            process_frame(silence_frame)
                        transcript = "".join(pieces).strip()
                        pieces.clear()
                        on_partial("")
                        if transcript:
                            on_final(transcript)
                finally:
                    events.task_done()

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
    r"désactiv\w*|desactiv\w*|coup\w*|éclair\w*|eclair\w*)\b",
    re.IGNORECASE,
)
_ACTION_TERM = re.compile(
    r"\b(?:allum\w*|étein\w*|etein\w*|activ\w*|désactiv\w*|desactiv\w*|"
    r"coup\w*|éclair\w*|eclair\w*)\b",
    re.IGNORECASE,
)
_ALFRED = re.compile(r"\balfred\b", re.IGNORECASE)
_HALOGENE_ASR_ALIASES = (
    re.compile(r"\bla\s+le\s+jeune\b", re.IGNORECASE),
    re.compile(r"\beug[eèé]ne\b", re.IGNORECASE),
)
_ETEINS_HALOGENE_ASR_ALIAS = re.compile(
    r"(?<![’'])\b(?:est|et|éteint|eteint)\s+la\s+(?:logement|lorraine)\b",
    re.IGNORECASE,
)
_CLIM_ASR_ALIAS_WITH_ARTICLE = re.compile(r"\ble\s+climat\b", re.IGNORECASE)
_CLIM_ASR_ALIAS = re.compile(r"\b(?:climat|clime)\b", re.IGNORECASE)
_SUPPORTED_TARGET = (
    r"(?=\s+(?:(?:le|la|les|du|des|l)\s+)?"
    r"(?:lumières?|halogène|salon|salle|luminaire|leds?|aides?|clim)\b)"
)
_ETEINS_ASR_ALIAS = re.compile(
    rf"(?<![’'])\b(?:est|et)\s+une?\b{_SUPPORTED_TARGET}",
    re.IGNORECASE,
)
_ETEINS_SPELLING_ALIAS = re.compile(
    rf"\b(?:éteint|eteint|éteindre|eteindre|étant|etant)\b{_SUPPORTED_TARGET}",
    re.IGNORECASE,
)
_ALLUME_ASR_ALIAS = re.compile(
    rf"\b(?:à|a)\s+l[’']?homme\b{_SUPPORTED_TARGET}",
    re.IGNORECASE,
)
_LEDS_ASR_ALIAS = re.compile(
    r"\b(?P<action>allum\w*|étein\w*|etein\w*)\s+les?\s+"
    r"(?:aides?|l[’']?aides?|laides?)\b",
    re.IGNORECASE,
)
_TURN_ON = re.compile(r"\b(?:allum\w*|activ\w*|éclair\w*|eclair\w*)\b", re.IGNORECASE)
_TURN_OFF = re.compile(
    r"\b(?:étein\w*|etein\w*|désactiv\w*|desactiv\w*|coup\w*)\b",
    re.IGNORECASE,
)
_EXPLICIT_TARGETS = (
    (re.compile(r"\bluminaire(?:\s+(?:du|de\s+le)\s+salon)?\b", re.IGNORECASE), "le luminaire du salon"),
    (re.compile(r"\bsalle\s+(?:à|a)\s+manger\b", re.IGNORECASE), "la salle à manger"),
    (re.compile(r"\bhalogène\b", re.IGNORECASE), "l'halogène"),
    (re.compile(r"\bleds?\b", re.IGNORECASE), "les leds"),
    (re.compile(r"\bclim(?:atisation|atiseur)?\b", re.IGNORECASE), "la clim"),
    (re.compile(r"\bsalon\b", re.IGNORECASE), "le salon"),
    (re.compile(r"\blumières?\b", re.IGNORECASE), "les lumières"),
)


class Terminal:
    def __init__(self, quiet: bool = False) -> None:
        self._lock = threading.Lock()
        self.quiet = quiet
        self._interactive = sys.stdout.isatty()
        self._partial = ""

    def partial(self, text: str) -> None:
        if self.quiet:
            return
        with self._lock:
            self._partial = text
            if self._interactive:
                print(f"\r\033[2K… {text}", end="", flush=True)
            elif text:
                print(f"[partiel] {text}", flush=True)

    def clear_partial(self) -> None:
        if self.quiet:
            return
        with self._lock:
            self._partial = ""
            if self._interactive:
                print("\r\033[2K", end="", flush=True)

    def line(self, text: str = "") -> None:
        if self.quiet:
            return
        with self._lock:
            if self._interactive:
                print("\r\033[2K", end="")
            print(text, flush=True)
            if self._interactive and self._partial:
                print(f"… {self._partial}", end="", flush=True)

    def warning(self, text: str) -> None:
        """Conserve les incidents utiles dans les logs, même en mode silencieux."""
        with self._lock:
            print(text, file=sys.stderr, flush=True)


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
    normalized = _ETEINS_HALOGENE_ASR_ALIAS.sub("éteins l'halogène", normalized)
    for alias in _HALOGENE_ASR_ALIASES:
        normalized = alias.sub("halogène", normalized)
    normalized = _CLIM_ASR_ALIAS_WITH_ARTICLE.sub("la clim", normalized)
    normalized = _CLIM_ASR_ALIAS.sub("clim", normalized)
    normalized = _ETEINS_ASR_ALIAS.sub("éteins", normalized)
    normalized = _ETEINS_SPELLING_ALIAS.sub("éteins", normalized)
    normalized = _ALLUME_ASR_ALIAS.sub("allume", normalized)
    normalized = _LEDS_ASR_ALIAS.sub(lambda match: f"{match.group('action')} les leds", normalized)
    return " ".join(normalized.split())


def transcript_for_intent(transcript: str) -> str:
    """Garde la commande après le mot-clé, géré séparément, pour stabiliser Laya."""
    keyword = list(_ALFRED.finditer(transcript))
    relevant = transcript[keyword[-1].end() :] if keyword else transcript
    return " ".join(relevant.split()).strip(" ,;:—-")


def deterministic_action(transcript: str) -> str | None:
    """Résout uniquement les commandes dont opération et cible sont explicites."""
    normalized = _SPACE_OR_PUNCTUATION.sub(" ", transcript.casefold()).strip()
    turns_on = bool(_TURN_ON.search(normalized))
    turns_off = bool(_TURN_OFF.search(normalized))
    if turns_on == turns_off:
        return None
    object_name = next(
        (name for pattern, name in _EXPLICIT_TARGETS if pattern.search(normalized)),
        None,
    )
    if object_name is None:
        return None
    operation = "allumer" if turns_on else "éteindre"
    return ACTION_BY_PARTS.get((operation, object_name))


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
    without_keyword = transcript_for_intent(normalized)
    if intent != "do_nothing" and not (
        _DIRECT_REQUEST.search(without_keyword)
        or (_POLITE_REQUEST.search(without_keyword) and _ACTION_TERM.search(without_keyword))
    ):
        return "commande directe non détectée"
    if intent != "do_nothing" and deterministic_action(transcript) is None:
        return "équipement pris en charge non détecté clairement"
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

    def wait_until_idle(self) -> None:
        """Attend les résultats déjà soumis, utile pour le mode texte interactif."""
        self._queue.join()

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
        result = self._engine.predict(transcript_for_intent(transcript))
        raw_confidence = result.confidence
        safety_reason = safety_override_reason(transcript, result.intent)
        explicit_action = deterministic_action(transcript)
        if safety_reason is None and explicit_action is not None:
            model_accepted = True
            candidate = explicit_action
        else:
            model_accepted = (
                safety_reason is None
                and (result.intent == "do_nothing" or raw_confidence >= self._threshold)
            )
            candidate = result.intent if model_accepted else "do_nothing"
        blocked_by_alfred = candidate != "do_nothing" and not alfred.allowed
        final_action = "do_nothing" if blocked_by_alfred else candidate

        self._terminal.line(f"🧠 Action proposée par Laya : {result.intent}")
        if explicit_action is not None and explicit_action != result.intent:
            self._terminal.line(
                f"🧭 Commande explicite retenue   : {explicit_action} (prioritaire sur Laya)"
            )
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


def run_kyutai_microphone(
    model_path: Path,
    worker: IntentWorker,
    terminal: Terminal,
    device: int | str | None = None,
    mic_gain: float = 1.0,
    *,
    silence_seconds: float = 0.32,
    speech_rms_threshold: float = 0.012,
    max_utterance_seconds: float = 15.0,
    audio_watchdog_timeout: float = 5.0,
    audio_retry_delay: float = 2.0,
) -> None:
    """Alimente Kyutai en continu et finalise rapidement chaque tour de parole."""
    import sounddevice as sd

    terminal.line("Chargement de Kyutai STT…")
    transcriber = KyutaiTranscriber(model_path)
    terminal.line(f"Kyutai STT chargé sur {transcriber.device}.")
    events: queue.Queue[KyutaiAudioEvent | None] = queue.Queue(maxsize=512)

    def partial(text: str) -> None:
        if text:
            terminal.partial(text)
        else:
            terminal.clear_partial()

    def final(text: str) -> None:
        terminal.clear_partial()
        terminal.line(f"🗣  Transcription : {text}")
        worker.submit(text)

    def transcribe_stream() -> None:
        try:
            transcriber.run_stream(events, partial, final)
        except Exception as error:
            terminal.warning(f"⚠ Échec du streaming Kyutai ({error}).")

    transcription_thread = threading.Thread(
        target=transcribe_stream,
        name="kyutai-transcriber",
        daemon=True,
    )
    worker.start()
    transcription_thread.start()
    try:
        while True:
            try:
                device_info: Any = sd.query_devices(device, "input")
                sample_rate = int(device_info["default_samplerate"])
                blocksize = int(sample_rate * 0.08)
                audio: queue.Queue[bytes] = queue.Queue(maxsize=128)
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
                terminal.line(f"Gain microphone logiciel : {mic_gain:g}×")
                terminal.line("Streaming Kyutai STT continu — Ctrl+C pour quitter.")
                speaking = False
                phrase_seconds = 0.0
                silent_seconds = 0.0
                with sd.RawInputStream(
                    samplerate=sample_rate,
                    blocksize=blocksize,
                    device=device,
                    dtype="int16",
                    channels=1,
                    latency="high",
                    callback=callback,
                ):
                    while True:
                        try:
                            chunk = audio.get(timeout=audio_watchdog_timeout)
                        except queue.Empty as error:
                            raise TimeoutError(
                                f"aucun bloc audio depuis {audio_watchdog_timeout:g} s"
                            ) from error
                        if mic_gain != 1.0:
                            samples = np.frombuffer(chunk, dtype=np.int16)
                            chunk = np.clip(
                                samples.astype(np.float32) * mic_gain, -32768, 32767
                            ).astype(np.int16).tobytes()
                        if dropped_blocks:
                            terminal.line(f"⚠ {dropped_blocks} bloc(s) audio perdu(s).")
                            dropped_blocks = 0

                        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(np.square(samples))) / 32768.0)
                        duration = len(samples) / sample_rate
                        starts_now = not speaking and rms >= speech_rms_threshold
                        if starts_now:
                            speaking = True
                            phrase_seconds = 0.0
                            silent_seconds = 0.0

                        try:
                            events.put_nowait(
                                KyutaiAudioEvent(
                                    chunk,
                                    sample_rate,
                                    start_of_utterance=starts_now,
                                )
                            )
                        except queue.Full:
                            terminal.warning(
                                "⚠ File de streaming Kyutai pleine : bloc audio ignoré."
                            )

                        if speaking:
                            phrase_seconds += duration
                            if rms >= speech_rms_threshold:
                                silent_seconds = 0.0
                            else:
                                silent_seconds += duration
                            reached_end = (
                                silent_seconds >= silence_seconds
                                or phrase_seconds >= max_utterance_seconds
                            )
                            if reached_end:
                                # Le marqueur est placé après l'audio déjà envoyé. Le thread
                                # d'inférence calcule alors le retard de 500 ms sans l'attendre.
                                try:
                                    events.put(
                                        KyutaiAudioEvent(
                                            b"",
                                            sample_rate,
                                            end_of_utterance=True,
                                        ),
                                        timeout=1.0,
                                    )
                                except queue.Full:
                                    terminal.warning(
                                        "⚠ File Kyutai saturée : fin de phrase retardée."
                                    )
                                speaking = False
                                phrase_seconds = 0.0
                                silent_seconds = 0.0
            except KeyboardInterrupt:
                terminal.clear_partial()
                terminal.line("Arrêt demandé.")
                break
            except (sd.PortAudioError, TimeoutError, ValueError) as error:
                terminal.warning(
                    f"⚠ Capture audio interrompue ({error}). Nouvelle tentative dans "
                    f"{audio_retry_delay:g} s."
                )
                time.sleep(audio_retry_delay)
    finally:
        try:
            events.put_nowait(None)
        except queue.Full:
            pass
        worker.close()


def run_microphone(
    vosk_model_path: Path,
    worker: IntentWorker,
    terminal: Terminal,
    device: int | str | None = None,
    mic_gain: float = 1.0,
    *,
    audio_watchdog_timeout: float = 5.0,
    audio_retry_delay: float = 2.0,
) -> None:
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model, SetLogLevel

    SetLogLevel(-1)
    vosk_model = Model(str(vosk_model_path))
    worker.start()
    try:
        while True:
            try:
                device_info: Any = sd.query_devices(device, "input")
                sample_rate = int(device_info["default_samplerate"])
                recognizer = KaldiRecognizer(vosk_model, sample_rate)
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
                terminal.line(f"Gain microphone logiciel : {mic_gain:g}×")
                terminal.line("Écoute continue — Ctrl+C pour quitter.")
                last_partial = ""
                with sd.RawInputStream(
                    samplerate=sample_rate,
                    blocksize=4000,
                    device=device,
                    dtype="int16",
                    channels=1,
                    latency="high",
                    callback=callback,
                ):
                    while True:
                        try:
                            chunk = audio.get(timeout=audio_watchdog_timeout)
                        except queue.Empty as error:
                            raise TimeoutError(
                                f"aucun bloc audio depuis {audio_watchdog_timeout:g} s"
                            ) from error
                        if mic_gain != 1.0:
                            samples = np.frombuffer(chunk, dtype=np.int16)
                            chunk = np.clip(
                                samples.astype(np.float32) * mic_gain, -32768, 32767
                            ).astype(np.int16).tobytes()
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
                            partial = str(
                                json.loads(recognizer.PartialResult()).get("partial", "")
                            ).strip()
                            if partial != last_partial:
                                terminal.partial(partial)
                                last_partial = partial
            except KeyboardInterrupt:
                terminal.clear_partial()
                terminal.line("Arrêt demandé.")
                break
            except (sd.PortAudioError, TimeoutError, ValueError) as error:
                terminal.clear_partial()
                terminal.warning(
                    f"⚠ Capture audio interrompue ({error}). Nouvelle tentative dans "
                    f"{audio_retry_delay:g} s."
                )
                time.sleep(audio_retry_delay)
    finally:
        worker.close()


def run_text_input(worker: IntentWorker, terminal: Terminal) -> None:
    """Boucle de debug interactive utilisant le même pipeline que les phrases Vosk finales."""
    terminal.line("Mode texte — saisissez une phrase puis Entrée. Ctrl+C ou Ctrl+D pour quitter.")
    worker.start()
    try:
        while True:
            try:
                transcript = input("" if getattr(terminal, "quiet", False) else "alfred> ").strip()
            except EOFError:
                break
            if not transcript:
                continue
            terminal.line(f"⌨️  Transcription : {transcript}")
            worker.submit(transcript)
            worker.wait_until_idle()
    except KeyboardInterrupt:
        terminal.line()
    finally:
        terminal.line("Arrêt demandé.")
        worker.close()
