from unittest.mock import patch

import pytest

from alfred_voice_assistant.models import (
    KYUTAI_STT_DIR_NAME,
    ensure_kyutai_stt_model,
    ensure_models,
    resolve_laya_backend,
)


def test_auto_backend_uses_mlx_on_apple_silicon() -> None:
    with (
        patch("alfred_voice_assistant.models.platform.system", return_value="Darwin"),
        patch("alfred_voice_assistant.models.platform.machine", return_value="arm64"),
    ):
        assert resolve_laya_backend() == "mlx"


def test_auto_backend_uses_torch_on_linux() -> None:
    with patch("alfred_voice_assistant.models.platform.system", return_value="Linux"):
        assert resolve_laya_backend() == "torch"


def test_mlx_is_rejected_on_linux() -> None:
    with patch("alfred_voice_assistant.models.platform.system", return_value="Linux"):
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            resolve_laya_backend("mlx")


def test_existing_kyutai_snapshot_is_reused(tmp_path) -> None:
    model = tmp_path / KYUTAI_STT_DIR_NAME
    model.mkdir()
    for filename in (
        "config.json",
        "mimi-pytorch-e351c8d8@125.safetensors",
        "model.safetensors",
        "tokenizer_en_fr_audio_8000.model",
    ):
        (model / filename).touch()

    assert ensure_kyutai_stt_model(tmp_path) == model


def test_ensure_models_selects_kyutai_by_default(tmp_path) -> None:
    with (
        patch("alfred_voice_assistant.models.ensure_kyutai_stt_model") as kyutai,
        patch("alfred_voice_assistant.models.ensure_vosk_model") as vosk,
        patch("alfred_voice_assistant.models.ensure_laya_model") as laya,
    ):
        ensure_models(tmp_path, "torch")

    kyutai.assert_called_once_with(tmp_path)
    vosk.assert_not_called()
    laya.assert_called_once_with(tmp_path, "torch")


def test_ensure_models_can_select_vosk(tmp_path) -> None:
    with (
        patch("alfred_voice_assistant.models.ensure_kyutai_stt_model") as kyutai,
        patch("alfred_voice_assistant.models.ensure_vosk_model") as vosk,
        patch("alfred_voice_assistant.models.ensure_laya_model") as laya,
    ):
        ensure_models(tmp_path, "torch", kyutai_stt=False)

    kyutai.assert_not_called()
    vosk.assert_called_once_with(tmp_path)
    laya.assert_called_once_with(tmp_path, "torch")
