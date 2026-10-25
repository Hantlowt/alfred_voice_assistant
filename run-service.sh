#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${ALFRED_CONFIG_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/alfred_voice_assistant/service.env}"

if [[ ! -r "$CONFIG_FILE" ]]; then
    echo "Configuration du service introuvable : $CONFIG_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

is_true() {
    case "$1" in
        1|true|TRUE|yes|YES|oui|OUI|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

ARGS=(
    --intent-mode "${ALFRED_INTENT_MODE:-flat}"
    --laya-backend "${ALFRED_LAYA_BACKEND:-auto}"
    --duplicate-window "${ALFRED_DUPLICATE_WINDOW:-3}"
)

if is_true "${ALFRED_REQUIRE_ALFRED:-true}"; then
    ARGS+=(--require-alfred)
else
    ARGS+=(--no-require-alfred)
fi
if is_true "${ALFRED_QUIET:-true}"; then
    ARGS+=(--quiet)
fi
if is_true "${ALFRED_EXECUTE_HOME_ASSISTANT:-true}"; then
    ARGS+=(--execute-home-assistant)
else
    ARGS+=(--no-execute-home-assistant)
fi
if is_true "${ALFRED_SPECIALIZATION:-true}"; then
    ARGS+=(--specialization)
else
    ARGS+=(--no-specialization)
fi

[[ -n "${HOME_ASSISTANT_URL:-}" ]] && ARGS+=(--home-assistant-url "$HOME_ASSISTANT_URL")
[[ -n "${HOME_ASSISTANT_TOKEN_FILE:-}" ]] && ARGS+=(--ha-token-file "$HOME_ASSISTANT_TOKEN_FILE")
[[ -n "${ALFRED_DEVICE:-}" ]] && ARGS+=(--device "$ALFRED_DEVICE")
[[ -n "${ALFRED_MODELS_DIR:-}" ]] && ARGS+=(--models-dir "$ALFRED_MODELS_DIR")
[[ -n "${ALFRED_THRESHOLD:-}" ]] && ARGS+=(--threshold "$ALFRED_THRESHOLD")

if is_true "${ALFRED_DRY_RUN:-false}"; then
    printf '%q ' "$SCRIPT_DIR/.venv/bin/alfred-local" "${ARGS[@]}"
    printf '\n'
    exit 0
fi

exec "$SCRIPT_DIR/.venv/bin/alfred-local" "${ARGS[@]}"
