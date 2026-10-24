"""Point d'entrée en ligne de commande."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .app import IntentWorker, Terminal, run_microphone
from .home_assistant import HomeAssistantExecutor, SimulationExecutor
from .intent import LayaIntentEngine
from .models import ensure_models, resolve_laya_backend


def _device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    default_models = Path(
        os.environ.get("ALFRED_MODELS_DIR", Path.home() / ".cache" / "alfred-voice-assistant")
    )
    default_token_file = Path(
        os.environ.get(
            "HOME_ASSISTANT_TOKEN_FILE",
            Path.home() / ".config" / "alfred_voice_assistant" / "home_assistant_token",
        )
    )
    parser = argparse.ArgumentParser(description="Assistant vocal local Vosk + Laya")
    parser.add_argument("--threshold", type=float, default=0.70, help="seuil de confiance (0 à 1)")
    parser.add_argument(
        "--duplicate-window",
        type=float,
        default=3.0,
        help="fenêtre anti-doublon en secondes",
    )
    parser.add_argument("--device", type=_device, help="index ou fragment du nom du microphone")
    parser.add_argument("--models-dir", type=Path, default=default_models)
    parser.add_argument("--list-devices", action="store_true", help="liste les périphériques audio")
    parser.add_argument(
        "--require-alfred",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="exige « alfred » dans la phrase courante ou la précédente",
    )
    parser.add_argument(
        "--intent-mode",
        choices=("flat", "factorized"),
        default="flat",
        help="un choix parmi les actions, ou deux choix opération + objet",
    )
    parser.add_argument(
        "--laya-backend",
        choices=("auto", "mlx", "torch"),
        default="auto",
        help="auto choisit MLX sur Apple Silicon et PyTorch CPU ailleurs",
    )
    parser.add_argument(
        "--execute-home-assistant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="exécute réellement les actions au lieu de les simuler",
    )
    parser.add_argument(
        "--home-assistant-url",
        default=os.environ.get("HOME_ASSISTANT_URL", "http://100.123.216.97:8123"),
    )
    parser.add_argument("--ha-token-file", type=Path, default=default_token_file)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold doit être compris entre 0 et 1.")
    if args.duplicate_window < 0:
        raise SystemExit("--duplicate-window doit être positif.")

    terminal = Terminal()
    backend = resolve_laya_backend(args.laya_backend)
    terminal.line(f"Modèles locaux : {args.models_dir.expanduser().resolve()}")
    terminal.line(
        "Mode Alfred obligatoire : " + ("ACTIVÉ" if args.require_alfred else "DÉSACTIVÉ")
    )
    terminal.line(f"Mode de décision Laya : {args.intent_mode}")
    terminal.line(f"Backend Laya : {backend}")
    if args.execute_home_assistant:
        executor = HomeAssistantExecutor(
            args.home_assistant_url,
            args.ha_token_file.expanduser(),
        )
        executor.check_connection()
        terminal.line(f"Home Assistant : CONNECTÉ ({args.home_assistant_url})")
    else:
        executor = SimulationExecutor()
        terminal.line("Home Assistant : simulation (aucun appel réel)")
    vosk_path, laya_path = ensure_models(args.models_dir.expanduser().resolve(), backend)
    engine = LayaIntentEngine(laya_path, mode=args.intent_mode, backend=backend)
    worker = IntentWorker(
        engine,
        terminal,
        args.threshold,
        args.duplicate_window,
        require_alfred=args.require_alfred,
        action_executor=executor,
    )
    run_microphone(vosk_path, worker, terminal, args.device)


if __name__ == "__main__":
    main()
