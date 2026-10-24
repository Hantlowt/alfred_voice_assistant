#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${ALFRED_REPO_URL:-https://github.com/Hantlowt/alfred_voice_assistant.git}"
INSTALL_DIR="${ALFRED_INSTALL_DIR:-$HOME/.local/share/alfred_voice_assistant}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/alfred_voice_assistant"
TOKEN_FILE="$CONFIG_DIR/home_assistant_token"
BIN_DIR="$HOME/.local/bin"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Ce script d'installation cible Linux (Lubuntu/Debian/Ubuntu)." >&2
    exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
    if [[ "$EUID" -eq 0 ]]; then
        APT=(apt-get)
    elif command -v sudo >/dev/null 2>&1; then
        APT=(sudo apt-get)
    else
        echo "sudo est requis pour installer PortAudio et les outils système." >&2
        exit 1
    fi
    "${APT[@]}" update
    "${APT[@]}" install -y ca-certificates curl git build-essential portaudio19-dev libasound2-dev
else
    echo "apt-get est introuvable : installez Git, curl, PortAudio et ALSA manuellement." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$(dirname "$INSTALL_DIR")"
if [[ -d "$INSTALL_DIR/.git" ]]; then
    git -C "$INSTALL_DIR" pull --ff-only
elif [[ -e "$INSTALL_DIR" ]]; then
    echo "Le chemin existe déjà sans être ce dépôt Git : $INSTALL_DIR" >&2
    exit 1
else
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

uv sync --project "$INSTALL_DIR" --locked --no-dev

mkdir -p "$CONFIG_DIR" "$BIN_DIR"
chmod 700 "$CONFIG_DIR"
if [[ ! -s "$TOKEN_FILE" ]]; then
    printf "Collez le token longue durée Home Assistant (saisie masquée) : " >/dev/tty
    IFS= read -r -s HA_TOKEN_INPUT </dev/tty
    printf "\n" >/dev/tty
    if [[ -z "$HA_TOKEN_INPUT" ]]; then
        echo "Token vide : installation interrompue." >&2
        exit 1
    fi
    printf '%s\n' "$HA_TOKEN_INPUT" >"$TOKEN_FILE"
    unset HA_TOKEN_INPUT
    chmod 600 "$TOKEN_FILE"
fi

if [[ -e "$BIN_DIR/alfred-local" && ! -L "$BIN_DIR/alfred-local" ]]; then
    echo "Impossible de créer le lanceur : $BIN_DIR/alfred-local existe déjà." >&2
    exit 1
fi
ln -sfn "$INSTALL_DIR/.venv/bin/alfred-local" "$BIN_DIR/alfred-local"

echo
echo "Installation terminée. Ajoutez $BIN_DIR au PATH si nécessaire."
echo "Lister les microphones : $BIN_DIR/alfred-local --list-devices"
echo "Démarrer en simulation : $BIN_DIR/alfred-local --device INDEX --require-alfred"
echo "Démarrer avec Home Assistant :"
echo "  $BIN_DIR/alfred-local --device INDEX --require-alfred --execute-home-assistant"
