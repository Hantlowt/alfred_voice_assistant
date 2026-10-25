#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/alfred_voice_assistant"
CONFIG_FILE="$CONFIG_DIR/service.env"
UNIT_NAME="alfred-voice-assistant.service"
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"
COMMAND="${1:-status}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "La gestion du service systemd est réservée à Linux." >&2
    exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
    echo "systemd est introuvable sur cette machine." >&2
    exit 1
fi

SUDO=()
if [[ "$EUID" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        echo "sudo est requis pour gérer le service système." >&2
        exit 1
    fi
    SUDO=(sudo)
fi

install_service() {
    mkdir -p "$CONFIG_DIR"
    chmod 700 "$CONFIG_DIR"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        install -m 600 "$SCRIPT_DIR/deploy/alfred.env.example" "$CONFIG_FILE"
        echo "Configuration créée : $CONFIG_FILE"
    fi

    local service_user service_group temp_unit
    service_user="$(id -un)"
    service_group="$(id -gn)"
    temp_unit="$(mktemp)"
    trap 'rm -f "$temp_unit"' RETURN
    {
        printf '%s\n' '[Unit]'
        printf '%s\n' 'Description=Assistant vocal local Alfred'
        printf '%s\n' 'Wants=network-online.target'
        printf '%s\n' 'After=network-online.target sound.target'
        printf '\n%s\n' '[Service]'
        printf '%s\n' 'Type=simple'
        printf 'User=%s\n' "$service_user"
        printf 'Group=%s\n' "$service_group"
        printf 'WorkingDirectory=%s\n' "$SCRIPT_DIR"
        printf 'Environment=HOME=%s\n' "$HOME"
        printf 'Environment=XDG_CONFIG_HOME=%s\n' "${XDG_CONFIG_HOME:-$HOME/.config}"
        printf 'ExecStart=%s/run-service.sh\n' "$SCRIPT_DIR"
        printf '%s\n' 'Restart=on-failure'
        printf '%s\n' 'RestartSec=5s'
        printf '%s\n' 'StandardOutput=journal'
        printf '%s\n' 'StandardError=journal'
        printf '\n%s\n' '[Install]'
        printf '%s\n' 'WantedBy=multi-user.target'
    } >"$temp_unit"
    "${SUDO[@]}" install -m 644 "$temp_unit" "$UNIT_PATH"
    "${SUDO[@]}" systemctl daemon-reload
    "${SUDO[@]}" systemctl enable "$UNIT_NAME"
    "${SUDO[@]}" systemctl restart "$UNIT_NAME"
    echo "Service installé, activé au démarrage et lancé."
}

case "$COMMAND" in
    install) install_service ;;
    start|stop|restart) "${SUDO[@]}" systemctl "$COMMAND" "$UNIT_NAME" ;;
    status) systemctl status "$UNIT_NAME" --no-pager ;;
    logs) "${SUDO[@]}" journalctl -u "$UNIT_NAME" -f ;;
    edit)
        "${EDITOR:-nano}" "$CONFIG_FILE"
        "${SUDO[@]}" systemctl restart "$UNIT_NAME"
        ;;
    *)
        echo "Usage : service.sh {install|start|stop|restart|status|logs|edit}" >&2
        exit 2
        ;;
esac
