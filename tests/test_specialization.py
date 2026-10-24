import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from alfred_voice_assistant.intent import ACTIONS
from alfred_voice_assistant.models import LAYA_REPO_ID
from alfred_voice_assistant.specialization import LatentSpecialization, bundled_specialization


def test_bundled_specialization_produces_all_action_probabilities() -> None:
    path = bundled_specialization("mlx")
    tensors = load_file(str(path / "vectors.safetensors"))
    dimension = tensors["transform"].shape[0]
    specialization = LatentSpecialization(
        path,
        backend="mlx",
        expected_labels=ACTIONS,
        expected_base_model=LAYA_REPO_ID,
    )

    result = specialization.predict(
        "alfred allume les lumières",
        lambda texts: np.ones((len(texts), dimension), dtype=np.float32),
    )

    assert result.choice in ACTIONS
    assert set(result.probabilities) == set(ACTIONS)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_specialization_rejects_the_wrong_backend() -> None:
    path = bundled_specialization("mlx")
    manifest = json.loads((path / "manifest.json").read_text())

    with pytest.raises(RuntimeError, match="incompatible"):
        LatentSpecialization(
            path,
            backend="torch",
            expected_labels=ACTIONS,
            expected_base_model=manifest["base_model"],
        )
