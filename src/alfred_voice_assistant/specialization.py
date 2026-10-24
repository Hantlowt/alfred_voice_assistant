"""Charge et applique un artefact latent léger exporté par Laya Studio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from safetensors.numpy import load_file


def bundled_specialization(backend: str) -> Path:
    name = "mlx" if backend == "mlx" else "torch"
    return Path(__file__).with_name("specializations") / name


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


@dataclass(frozen=True, slots=True)
class SpecializedPrediction:
    choice: str
    probabilities: dict[str, float]


class LatentSpecialization:
    """Runtime minimal du format `laya-specialization` de Laya Studio."""

    def __init__(
        self,
        path: Path,
        *,
        backend: str,
        expected_labels: Sequence[str],
        expected_base_model: str,
    ) -> None:
        self.path = path
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            payload = json.loads((path / "specialization.json").read_text(encoding="utf-8"))
            arrays = load_file(str(path / "vectors.safetensors"))
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError(f"Artefact Laya Studio illisible : {path}") from error

        runtime_backend = "pytorch" if backend == "torch" else backend
        if manifest.get("format") != "laya-specialization" or manifest.get("format_version") != 1:
            raise RuntimeError("Format d'artefact Laya Studio non pris en charge.")
        if runtime_backend not in manifest.get("backend_compatibility", []):
            raise RuntimeError(
                f"Artefact incompatible avec {backend} : "
                f"{manifest.get('backend_compatibility', [])}"
            )
        if manifest.get("base_model") != expected_base_model:
            raise RuntimeError(
                "L'artefact et le modèle Laya de base ne correspondent pas : "
                f"{manifest.get('base_model')} != {expected_base_model}"
            )

        artifact = payload.get("artifact", {})
        if artifact.get("strategy") != "residual_embedding_transform":
            raise RuntimeError("Cette version attend une transformation résiduelle Laya Studio.")
        self.labels = tuple(artifact.get("labels", ()))
        if self.labels != tuple(expected_labels):
            raise RuntimeError("Les actions de l'artefact Laya Studio ne correspondent pas au code.")
        try:
            self._transform = np.asarray(arrays["transform"], dtype=np.float32)
            self._prototypes = np.asarray(arrays["prototypes"], dtype=np.float32)
            self._temperature = float(artifact["params"]["temperature"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Transformation résiduelle Laya Studio incomplète.") from error
        if self._transform.ndim != 2 or self._transform.shape[0] != self._transform.shape[1]:
            raise RuntimeError("Matrice de transformation Laya Studio invalide.")
        if self._prototypes.shape != (len(self.labels), self._transform.shape[0]):
            raise RuntimeError("Prototypes Laya Studio incompatibles avec la transformation.")
        if self._temperature <= 0:
            raise RuntimeError("Température Laya Studio invalide.")

        self.base_revision = str(manifest.get("base_revision") or "inconnue")
        self.strategy = str(manifest["strategy"])

    def predict(
        self,
        transcript: str,
        embed: Callable[[Sequence[str]], np.ndarray],
    ) -> SpecializedPrediction:
        encoded = _normalize(np.asarray(embed([transcript]), dtype=np.float32))
        if encoded.shape != (1, self._transform.shape[0]):
            raise RuntimeError(
                "Dimension d'embedding incompatible avec l'artefact Laya Studio : "
                f"{encoded.shape}"
            )
        transformed = _normalize(encoded @ self._transform)
        logits = transformed @ self._prototypes.T
        scaled = logits[0] / self._temperature
        scaled -= scaled.max()
        probabilities_array = np.exp(scaled)
        probabilities_array /= probabilities_array.sum()
        probabilities = {
            label: float(probability)
            for label, probability in zip(self.labels, probabilities_array, strict=True)
        }
        choice = self.labels[int(np.argmax(probabilities_array))]
        return SpecializedPrediction(choice, probabilities)
