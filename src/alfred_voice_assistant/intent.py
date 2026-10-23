"""Interface du moteur d'intentions et implémentation Laya MLX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

ACTIONS = (
    "turn_on_light",
    "turn_off_light",
    "get_weather",
    "play_music",
    "stop_music",
    "do_nothing",
)

INTENT_QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": (
            "Choisis l'unique action explicitement demandée à l'assistant dans la phrase française. "
            "Une commande doit être actuelle, directe et sembler adressée à l'assistant. Ignore les "
            "conversations ordinaires, descriptions, questions rhétoriques, citations et propos "
            "rapportés (par exemple « il a dit d'allumer »). Ignore aussi toute demande niée ou "
            "annulée (par exemple « n'allume pas », « ne joue rien »). N'infère jamais une action "
            "implicite. Si plusieurs actions se contredisent, si l'adresse à l'assistant n'est pas "
            "claire, ou au moindre doute, choisis do_nothing."
        ),
        "criteria": {
            "turn_on_light": "ordre direct et affirmatif d'allumer la lumière",
            "turn_off_light": "ordre direct et affirmatif d'éteindre la lumière",
            "get_weather": "demande directe de météo ou de prévisions météorologiques",
            "play_music": "ordre direct et affirmatif de lancer ou reprendre de la musique",
            "stop_music": "ordre direct et affirmatif d'arrêter ou mettre en pause la musique",
            "do_nothing": (
                "aucune commande sûre : conversation, propos rapporté, négation, ambiguïté, "
                "phrase non adressée à l'assistant ou action non prise en charge"
            ),
        },
    }
}


@dataclass(frozen=True, slots=True)
class IntentResult:
    intent: str
    probabilities: Mapping[str, float]
    latency_ms: float


class IntentEngine(Protocol):
    """Point d'extension pour un modèle Laya Studio spécialisé."""

    def predict(self, transcript: str) -> IntentResult: ...


class LayaMlxIntentEngine:
    """Adaptateur minimal autour de l'API publique de laya-mlx."""

    def __init__(self, model_path: Path) -> None:
        import laya_mlx as laya

        print("Chargement de Laya MLX en mémoire…")
        self._agent = laya.load(str(model_path), dtype="float16")

    def predict(self, transcript: str) -> IntentResult:
        from time import perf_counter

        started = perf_counter()
        response = self._agent.predict(transcript, INTENT_QUESTION)
        latency_ms = (perf_counter() - started) * 1000
        answer = response["answers"]["intent"]
        probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
        intent = str(answer["choice"])
        if intent not in ACTIONS or set(probabilities) != set(ACTIONS):
            raise RuntimeError("Réponse Laya incompatible avec les actions configurées.")
        return IntentResult(intent=intent, probabilities=probabilities, latency_ms=latency_ms)
