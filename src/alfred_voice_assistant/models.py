"""Téléchargement initial et validation des deux modèles locaux."""

from __future__ import annotations

import platform
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

VOSK_MODEL_NAME = "vosk-model-small-fr-0.22"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
LAYA_REPO_ID = "aac6fef/laya-multilingual-mlx"
LAYA_DIR_NAME = "laya-multilingual-mlx"
LAYA_TORCH_REPO_ID = "convaiinnovations/laya"
LAYA_TORCH_DIR_NAME = "laya-multilingual-pytorch"


def _valid_vosk_model(path: Path) -> bool:
    return (path / "am" / "final.mdl").is_file() and (path / "conf" / "mfcc.conf").is_file()


def _valid_laya_model(path: Path) -> bool:
    return (path / "mlx_config.json").is_file() and any(path.glob("*.safetensors"))


def _valid_laya_torch_model(path: Path) -> bool:
    checkpoint = path / "multilingual"
    return (checkpoint / "rl_agent_config.json").is_file() and (
        checkpoint / "model.safetensors"
    ).is_file()


def resolve_laya_backend(requested: str = "auto") -> str:
    if requested == "auto":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return "mlx"
        return "torch"
    if requested not in {"mlx", "torch"}:
        raise ValueError(f"Backend Laya inconnu : {requested}")
    if requested == "mlx" and not (
        platform.system() == "Darwin" and platform.machine() == "arm64"
    ):
        raise RuntimeError("Le backend MLX requiert un Mac Apple Silicon.")
    return requested


def _download(url: str, destination: Path) -> None:
    last_percent = -1

    def report(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_percent
        if total > 0:
            percent = min(100, blocks * block_size * 100 // total)
            if percent != last_percent:
                print(f"\r  téléchargement : {percent:3d}%", end="", flush=True)
                last_percent = percent

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print(file=sys.stdout)


def ensure_vosk_model(models_dir: Path) -> Path:
    """Retourne un modèle Vosk local, en le téléchargeant atomiquement si nécessaire."""
    destination = models_dir / VOSK_MODEL_NAME
    if _valid_vosk_model(destination):
        return destination
    if destination.exists():
        raise RuntimeError(f"Modèle Vosk incomplet : supprimez {destination} puis relancez.")

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Modèle Vosk absent, téléchargement unique de {VOSK_MODEL_NAME}…")
    with tempfile.TemporaryDirectory(prefix="vosk-", dir=models_dir) as temp_name:
        temp_dir = Path(temp_name)
        archive = temp_dir / "model.zip"
        _download(VOSK_MODEL_URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            root = temp_dir.resolve()
            for member in zipped.infolist():
                target = (temp_dir / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError("Archive Vosk non sûre (chemin hors destination).")
            zipped.extractall(temp_dir)
        extracted = temp_dir / VOSK_MODEL_NAME
        if not _valid_vosk_model(extracted):
            raise RuntimeError("Le modèle Vosk téléchargé est incomplet.")
        shutil.move(str(extracted), destination)
    return destination


def ensure_laya_model(models_dir: Path, backend: str = "auto") -> Path:
    """Retourne un snapshot Laya local, en le téléchargeant atomiquement si nécessaire."""
    backend = resolve_laya_backend(backend)
    if backend == "torch":
        return ensure_laya_torch_model(models_dir)

    destination = models_dir / LAYA_DIR_NAME
    if _valid_laya_model(destination):
        return destination
    if destination.exists():
        raise RuntimeError(f"Modèle Laya incomplet : supprimez {destination} puis relancez.")

    from huggingface_hub import snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Modèle Laya absent, téléchargement unique de {LAYA_REPO_ID}…")
    with tempfile.TemporaryDirectory(prefix="laya-", dir=models_dir) as temp_name:
        temp_dir = Path(temp_name) / "snapshot"
        snapshot_download(repo_id=LAYA_REPO_ID, local_dir=temp_dir)
        if not _valid_laya_model(temp_dir):
            raise RuntimeError("Le modèle Laya téléchargé est incomplet.")
        shutil.move(str(temp_dir), destination)
    return destination


def ensure_laya_torch_model(models_dir: Path) -> Path:
    """Télécharge le checkpoint multilingue PyTorch utilisé sur Linux/CPU."""
    destination = models_dir / LAYA_TORCH_DIR_NAME
    if _valid_laya_torch_model(destination):
        return destination
    if destination.exists():
        raise RuntimeError(f"Modèle Laya PyTorch incomplet : supprimez {destination} puis relancez.")

    from huggingface_hub import snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Modèle Laya PyTorch absent, téléchargement unique de {LAYA_TORCH_REPO_ID}…")
    with tempfile.TemporaryDirectory(prefix="laya-torch-", dir=models_dir) as temp_name:
        temp_dir = Path(temp_name) / "snapshot"
        snapshot_download(
            repo_id=LAYA_TORCH_REPO_ID,
            local_dir=temp_dir,
            allow_patterns=["multilingual/**"],
        )
        if not _valid_laya_torch_model(temp_dir):
            raise RuntimeError("Le modèle Laya PyTorch téléchargé est incomplet.")
        shutil.move(str(temp_dir), destination)
    return destination


def ensure_models(models_dir: Path, backend: str = "auto") -> tuple[Path, Path]:
    """Prépare tous les poids avant de démarrer le microphone."""
    return ensure_vosk_model(models_dir), ensure_laya_model(models_dir, backend)
