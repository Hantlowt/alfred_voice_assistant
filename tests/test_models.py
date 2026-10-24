from unittest.mock import patch

import pytest

from alfred_voice_assistant.models import resolve_laya_backend


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
