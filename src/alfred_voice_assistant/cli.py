"""Point d'entrée en ligne de commande."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

from .app import IntentWorker, Terminal, run_microphone, run_text_input
from .home_assistant import HomeAssistantExecutor, SimulationExecutor
from .intent import LayaIntentEngine
from .models import ensure_laya_model, ensure_models, resolve_laya_backend
from .specialization import bundled_specialization


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="seuil de confiance entre 0 et 1 (désactivé par défaut)",
    )
    parser.add_argument(
        "--duplicate-window",
        type=float,
        default=3.0,
        help="fenêtre anti-doublon en secondes",
    )
    parser.add_argument("--device", type=_device, help="index ou fragment du nom du microphone")
    parser.add_argument(
        "--mic-gain",
        type=float,
        default=1.0,
        help="gain logiciel appliqué avant Vosk (1.0 = aucun gain)",
    )
    parser.add_argument("--models-dir", type=Path, default=default_models)
    parser.add_argument("--list-devices", action="store_true", help="liste les périphériques audio")
    parser.add_argument(
        "--text-mode",
        "--text",
        action="store_true",
        help="lit les phrases au clavier au lieu d'utiliser Vosk et le microphone",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="n'affiche aucune sortie normale (les erreurs fatales restent visibles)",
    )
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
        "--specialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="utilise l'artefact Laya Studio fourni en mode flat",
    )
    parser.add_argument(
        "--specialization-dir",
        type=Path,
        help="artefact Laya Studio personnalisé (manifest.json + vectors.safetensors)",
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

        if not args.quiet:
            print(sd.query_devices())
        return
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold doit être compris entre 0 et 1.")
    if args.duplicate_window < 0:
        raise SystemExit("--duplicate-window doit être positif.")
    if not 0 < args.mic_gain <= 10:
        raise SystemExit("--mic-gain doit être supérieur à 0 et inférieur ou égal à 10.")

    terminal = Terminal(quiet=args.quiet)
    backend = resolve_laya_backend(args.laya_backend)
    use_specialization = args.specialization and args.intent_mode == "flat"
    specialization_path = None
    if use_specialization:
        specialization_path = (
            args.specialization_dir.expanduser().resolve()
            if args.specialization_dir
            else bundled_specialization(backend)
        )
    threshold = args.threshold if args.threshold is not None else 0.0
    terminal.line(f"Modèles locaux : {args.models_dir.expanduser().resolve()}")
    terminal.line(
        "Mode Alfred obligatoire : " + ("ACTIVÉ" if args.require_alfred else "DÉSACTIVÉ")
    )
    terminal.line(f"Mode de décision Laya : {args.intent_mode}")
    terminal.line("Entrée : " + ("TEXTE" if args.text_mode else "MICROPHONE"))
    terminal.line(f"Backend Laya : {backend}")
    if specialization_path is not None:
        terminal.line(f"Spécialisation Laya Studio : {specialization_path}")
    elif args.intent_mode == "factorized" and args.specialization:
        terminal.line("Spécialisation Laya Studio : non applicable au mode factorized")
    else:
        terminal.line("Spécialisation Laya Studio : DÉSACTIVÉE")
    terminal.line(
        "Seuil de confiance : "
        + (f"{threshold:.0%}" if threshold > 0 else "DÉSACTIVÉ")
    )
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
    models_dir = args.models_dir.expanduser().resolve()
    with contextlib.ExitStack() as stack:
        if args.quiet:
            sink = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
            stack.enter_context(contextlib.redirect_stdout(sink))
            stack.enter_context(contextlib.redirect_stderr(sink))
        if args.text_mode:
            vosk_path = None
            laya_path = ensure_laya_model(models_dir, backend)
        else:
            vosk_path, laya_path = ensure_models(models_dir, backend)
        engine = LayaIntentEngine(
            laya_path,
            mode=args.intent_mode,
            backend=backend,
            specialization_path=specialization_path,
        )
    worker = IntentWorker(
        engine,
        terminal,
        threshold,
        args.duplicate_window,
        require_alfred=args.require_alfred,
        action_executor=executor,
    )
    if args.text_mode:
        run_text_input(worker, terminal)
    else:
        assert vosk_path is not None
        run_microphone(vosk_path, worker, terminal, args.device, args.mic_gain)


if __name__ == "__main__":
    main()
