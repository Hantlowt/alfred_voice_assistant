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
LAYA_REVISION = "f2b4faf51023039425946074e2cf1361d2db11d5"
LAYA_DIR_NAME = "laya-multilingual-mlx"
LAYA_TORCH_REPO_ID = "convaiinnovations/laya-multilingual"
LAYA_TORCH_REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"
LAYA_TORCH_DIR_NAME = "laya-multilingual-pytorch-v2"
KYUTAI_STT_REPO_ID = "kyutai/stt-1b-en_fr"
KYUTAI_STT_REVISION = "1c34c6b4f7e9299bb61985f145052ff131005dde"
KYUTAI_STT_DIR_NAME = "kyutai-stt-1b-en_fr"


def _valid_vosk_model(path: Path) -> bool:
    return (path / "am" / "final.mdl").is_file() and (path / "conf" / "mfcc.conf").is_file()


def _valid_laya_model(path: Path) -> bool:
    return (path / "mlx_config.json").is_file() and any(path.glob("*.safetensors"))


def _valid_laya_torch_model(path: Path) -> bool:
    return (path / "rl_agent_config.json").is_file() and (path / "model.safetensors").is_file()


def _valid_kyutai_stt_model(path: Path) -> bool:
    return all(
        (path / filename).is_file()
        for filename in (
            "config.json",
            "mimi-pytorch-e351c8d8@125.safetensors",
            "model.safetensors",
            "tokenizer_en_fr_audio_8000.model",
        )
    )


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
        snapshot_download(repo_id=LAYA_REPO_ID, revision=LAYA_REVISION, local_dir=temp_dir)
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
            revision=LAYA_TORCH_REVISION,
            local_dir=temp_dir,
        )
        if not _valid_laya_torch_model(temp_dir):
            raise RuntimeError("Le modèle Laya PyTorch téléchargé est incomplet.")
        shutil.move(str(temp_dir), destination)
    return destination


def ensure_kyutai_stt_model(models_dir: Path) -> Path:
    """Télécharge atomiquement le modèle de transcription français/anglais Kyutai."""
    destination = models_dir / KYUTAI_STT_DIR_NAME
    if _valid_kyutai_stt_model(destination):
        return destination
    if destination.exists():
        raise RuntimeError(f"Modèle Kyutai STT incomplet : supprimez {destination} puis relancez.")

    from huggingface_hub import snapshot_download

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Modèle Kyutai STT absent, téléchargement unique de {KYUTAI_STT_REPO_ID}…")
    with tempfile.TemporaryDirectory(prefix="kyutai-stt-", dir=models_dir) as temp_name:
        temp_dir = Path(temp_name) / "snapshot"
        snapshot_download(
            repo_id=KYUTAI_STT_REPO_ID,
            revision=KYUTAI_STT_REVISION,
            local_dir=temp_dir,
        )
        if not _valid_kyutai_stt_model(temp_dir):
            raise RuntimeError("Le modèle Kyutai STT téléchargé est incomplet.")
        shutil.move(str(temp_dir), destination)
    return destination


def ensure_models(
    models_dir: Path,
    backend: str = "auto",
    *,
    kyutai_stt: bool = True,
) -> tuple[Path, Path]:
    """Prépare tous les poids avant de démarrer le microphone."""
    asr_path = (
        ensure_kyutai_stt_model(models_dir)
        if kyutai_stt
        else ensure_vosk_model(models_dir)
    )
    return asr_path, ensure_laya_model(models_dir, backend)
