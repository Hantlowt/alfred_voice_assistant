#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${ALFRED_REPO_URL:-https://github.com/Hantlowt/alfred_voice_assistant.git}"
INSTALL_DIR="${ALFRED_INSTALL_DIR:-$HOME/.local/share/alfred_voice_assistant}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/alfred_voice_assistant"
TOKEN_FILE="$CONFIG_DIR/home_assistant_token"
SERVICE_CONFIG="$CONFIG_DIR/service.env"
BIN_DIR="$HOME/.local/bin"
MODE="${1:-install}"

if [[ "$MODE" != "install" && "$MODE" != "--update" ]]; then
    echo "Usage : install.sh [--update]" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Ce script d'installation cible Linux (Lubuntu/Debian/Ubuntu)." >&2
    exit 1
fi

if [[ "$MODE" == "install" ]] && command -v apt-get >/dev/null 2>&1; then
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
elif [[ "$MODE" == "install" ]]; then
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
fi
chmod 600 "$TOKEN_FILE"

if [[ ! -f "$SERVICE_CONFIG" ]]; then
    DEFAULT_HA_URL="${HOME_ASSISTANT_URL:-http://homeassistant.local:8123}"
    echo "Microphones détectés :"
    "$INSTALL_DIR/.venv/bin/alfred-local" --list-devices || \
        echo "Impossible de lister les microphones maintenant ; vous pourrez configurer le nom plus tard." >&2
    printf "Adresse Home Assistant [%s] : " "$DEFAULT_HA_URL" >/dev/tty
    IFS= read -r HA_URL_INPUT </dev/tty
    HA_URL_INPUT="${HA_URL_INPUT:-$DEFAULT_HA_URL}"
    printf "Microphone (index ou fragment du nom, vide = périphérique par défaut) : " >/dev/tty
    IFS= read -r ALFRED_DEVICE_INPUT </dev/tty
    {
        printf 'HOME_ASSISTANT_URL=%q\n' "$HA_URL_INPUT"
        printf 'HOME_ASSISTANT_TOKEN_FILE=%q\n' "$TOKEN_FILE"
        printf 'ALFRED_DEVICE=%q\n' "$ALFRED_DEVICE_INPUT"
        printf '%s\n' 'ALFRED_MIC_GAIN=1.0'
        printf 'ALFRED_MODELS_DIR=%q\n' "$HOME/.cache/alfred-voice-assistant"
        printf '%s\n' 'ALFRED_REQUIRE_ALFRED=true'
        printf '%s\n' 'ALFRED_QUIET=true'
        printf '%s\n' 'ALFRED_EXECUTE_HOME_ASSISTANT=true'
        printf '%s\n' 'ALFRED_SPECIALIZATION=true'
        printf '%s\n' 'ALFRED_INTENT_MODE=flat'
        printf '%s\n' 'ALFRED_LAYA_BACKEND=auto'
        printf '%s\n' 'ALFRED_DUPLICATE_WINDOW=3'
        printf '%s\n' 'ALFRED_THRESHOLD='
    } >"$SERVICE_CONFIG"
    chmod 600 "$SERVICE_CONFIG"
    unset HA_URL_INPUT ALFRED_DEVICE_INPUT
fi

for launcher in alfred-local alfred-update alfred-service; do
    if [[ -e "$BIN_DIR/$launcher" && ! -L "$BIN_DIR/$launcher" ]]; then
        echo "Impossible de créer le lanceur : $BIN_DIR/$launcher existe déjà." >&2
        exit 1
    fi
done
ln -sfn "$INSTALL_DIR/.venv/bin/alfred-local" "$BIN_DIR/alfred-local"
ln -sfn "$INSTALL_DIR/update.sh" "$BIN_DIR/alfred-update"
ln -sfn "$INSTALL_DIR/service.sh" "$BIN_DIR/alfred-service"

if command -v systemctl >/dev/null 2>&1; then
    "$INSTALL_DIR/service.sh" install
else
    echo "Avertissement : systemd absent, service automatique non installé." >&2
fi

echo
echo "Installation/mise à jour terminée. Ajoutez $BIN_DIR au PATH si nécessaire."
echo "Mettre à jour plus tard : $BIN_DIR/alfred-update"
echo "Configurer le service : $BIN_DIR/alfred-service edit"
echo "État du service : $BIN_DIR/alfred-service status"
echo "Logs du service : $BIN_DIR/alfred-service logs"
echo "Lister les microphones : $BIN_DIR/alfred-local --list-devices"
echo "Démarrer en simulation : $BIN_DIR/alfred-local --device INDEX --require-alfred"
echo "Démarrer avec Home Assistant :"
echo "  $BIN_DIR/alfred-local --device INDEX --require-alfred --execute-home-assistant"
