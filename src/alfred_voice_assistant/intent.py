"""Interface du moteur d'intentions et implémentation Laya MLX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

ACTIONS = (
    "do_nothing",
    "éteins les lumières",
    "allume les lumières",
    "allume l'halogène",
    "allume le salon",
    "allume la salle à manger",
    "éteins l'halogène",
    "éteins le luminaire du salon",
    "éteins le salon",
    "éteins la salle à manger",
    "allume les leds",
    "éteins les leds",
    "allume la clim",
    "éteins la clim",
)

INTENT_QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": (
            "Choisis l'unique action domotique explicitement demandée dans la phrase française. "
            "Une commande doit être actuelle, directe et adressée à l'assistant. Ignore les "
            "conversations ordinaires, descriptions, questions rhétoriques, citations et propos "
            "rapportés (par exemple « il a dit d'allumer »). Ignore aussi toute demande niée ou "
            "annulée (par exemple « n'allume pas », « n'éteins pas »). N'infère jamais une action "
            "implicite. Si plusieurs actions se contredisent, si l'adresse à l'assistant n'est pas "
            "claire, ou au moindre doute, choisis do_nothing."
        ),
        "criteria": {
            "do_nothing": (
                "aucune commande sûre : conversation, propos rapporté, négation, ambiguïté, "
                "phrase non adressée à l'assistant ou action non prise en charge"
            ),
            "éteins les lumières": "éteindre toutes les lumières, sans zone particulière",
            "allume les lumières": "allumer toutes les lumières, sans zone particulière",
            "allume l'halogène": "allumer spécifiquement la lampe halogène",
            "allume le salon": "allumer les éclairages de la zone salon",
            "allume la salle à manger": "allumer les éclairages de la salle à manger",
            "éteins l'halogène": "éteindre spécifiquement la lampe halogène",
            "éteins le luminaire du salon": "éteindre seulement le luminaire du salon",
            "éteins le salon": "éteindre tous les éclairages de la zone salon",
            "éteins la salle à manger": "éteindre les éclairages de la salle à manger",
            "allume les leds": "allumer les rubans ou éclairages LED",
            "éteins les leds": "éteindre les rubans ou éclairages LED",
            "allume la clim": "allumer ou démarrer la climatisation",
            "éteins la clim": "éteindre ou arrêter la climatisation",
        },
    }
}

OPERATION_QUESTION = {
    "operation": {
        "type": "choice",
        "instructions": (
            "Détermine uniquement si la phrase contient une demande actuelle, directe et affirmative "
            "d'allumer ou d'éteindre un équipement. Ignore les conversations, propos rapportés, "
            "négations et demandes ambiguës. Au moindre doute, choisis do_nothing."
        ),
        "criteria": {
            "allumer": "ordre direct d'allumer, activer ou démarrer un équipement",
            "éteindre": "ordre direct d'éteindre, désactiver ou arrêter un équipement",
            "do_nothing": "aucun ordre direct et affirmatif d'allumer ou d'éteindre",
        },
    }
}

OBJECT_QUESTION = {
    "objet": {
        "type": "choice",
        "instructions": (
            "Identifie uniquement l'équipement explicitement visé par la commande française. "
            "Distingue les lumières générales, la zone salon et le luminaire spécifique du salon. "
            "Si aucun équipement de la liste n'est demandé clairement, choisis do_nothing."
        ),
        "criteria": {
            "les lumières": "toutes les lumières, sans pièce ni luminaire spécifique",
            "l'halogène": "la lampe halogène",
            "le salon": "tous les éclairages de la zone salon",
            "la salle à manger": "les éclairages de la salle à manger",
            "le luminaire du salon": "le luminaire spécifique situé dans le salon",
            "les leds": "les rubans ou éclairages LED",
            "la clim": "la climatisation",
            "do_nothing": "aucun équipement pris en charge n'est explicitement visé",
        },
    }
}

ACTION_BY_PARTS = {
    ("allumer", "les lumières"): "allume les lumières",
    ("allumer", "l'halogène"): "allume l'halogène",
    ("allumer", "le salon"): "allume le salon",
    ("allumer", "la salle à manger"): "allume la salle à manger",
    ("allumer", "les leds"): "allume les leds",
    ("allumer", "la clim"): "allume la clim",
    ("éteindre", "les lumières"): "éteins les lumières",
    ("éteindre", "l'halogène"): "éteins l'halogène",
    ("éteindre", "le salon"): "éteins le salon",
    ("éteindre", "la salle à manger"): "éteins la salle à manger",
    ("éteindre", "le luminaire du salon"): "éteins le luminaire du salon",
    ("éteindre", "les leds"): "éteins les leds",
    ("éteindre", "la clim"): "éteins la clim",
}


@dataclass(frozen=True, slots=True)
class ProbabilityReport:
    label: str
    probabilities: Mapping[str, float]
    latency_ms: float


@dataclass(frozen=True, slots=True)
class IntentResult:
    intent: str
    confidence: float
    reports: tuple[ProbabilityReport, ...]
    latency_ms: float


class IntentEngine(Protocol):
    """Point d'extension pour un modèle Laya Studio spécialisé."""

    def predict(self, transcript: str) -> IntentResult: ...


class LayaIntentEngine:
    """Adaptateur commun aux API publiques Laya MLX et Laya PyTorch."""

    def __init__(self, model_path: Path, mode: str = "flat", backend: str = "mlx") -> None:
        if mode not in {"flat", "factorized"}:
            raise ValueError(f"Mode d'intention inconnu : {mode}")
        if backend == "mlx":
            import laya_mlx as laya

            print("Chargement de Laya MLX en mémoire…")
            self._agent = laya.load(str(model_path), dtype="float16")
        elif backend == "torch":
            import laya

            print("Chargement de Laya PyTorch sur CPU en mémoire…")
            self._agent = laya.load(str(model_path), subfolder="multilingual", device="cpu")
        else:
            raise ValueError(f"Backend Laya inconnu : {backend}")
        self._mode = mode

    def predict(self, transcript: str) -> IntentResult:
        from time import perf_counter

        if self._mode == "factorized":
            return self._predict_factorized(transcript)

        started = perf_counter()
        response = self._agent.predict(transcript, INTENT_QUESTION)
        latency_ms = (perf_counter() - started) * 1000
        answer = response["answers"]["intent"]
        probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
        intent = str(answer["choice"])
        if intent not in ACTIONS or set(probabilities) != set(ACTIONS):
            raise RuntimeError("Réponse Laya incompatible avec les actions configurées.")
        return IntentResult(
            intent=intent,
            confidence=probabilities[intent],
            reports=(ProbabilityReport("actions", probabilities, latency_ms),),
            latency_ms=latency_ms,
        )

    def _predict_factorized(self, transcript: str) -> IntentResult:
        operation, operation_report = self._predict_one(
            transcript, OPERATION_QUESTION, "operation", "opération"
        )
        object_name, object_report = self._predict_one(
            transcript, OBJECT_QUESTION, "objet", "objet"
        )
        intent = ACTION_BY_PARTS.get((operation, object_name), "do_nothing")
        if operation == "do_nothing":
            confidence = operation_report.probabilities[operation]
        else:
            confidence = min(
                operation_report.probabilities[operation],
                object_report.probabilities[object_name],
            )
        return IntentResult(
            intent=intent,
            confidence=confidence,
            reports=(operation_report, object_report),
            latency_ms=operation_report.latency_ms + object_report.latency_ms,
        )

    def _predict_one(
        self,
        transcript: str,
        question: dict[str, dict[str, object]],
        answer_key: str,
        report_label: str,
    ) -> tuple[str, ProbabilityReport]:
        from time import perf_counter

        started = perf_counter()
        response = self._agent.predict(transcript, question)
        latency_ms = (perf_counter() - started) * 1000
        answer = response["answers"][answer_key]
        probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
        choice = str(answer["choice"])
        if choice not in probabilities:
            raise RuntimeError(f"Réponse Laya invalide pour {report_label}.")
        return choice, ProbabilityReport(report_label, probabilities, latency_ms)


# Compatibilité pour les imports existants du prototype.
LayaMlxIntentEngine = LayaIntentEngine
