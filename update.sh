#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
while [[ -L "$SCRIPT_SOURCE" ]]; do
    LINK_TARGET="$(readlink "$SCRIPT_SOURCE")"
    if [[ "$LINK_TARGET" = /* ]]; then
        SCRIPT_SOURCE="$LINK_TARGET"
    else
        SCRIPT_SOURCE="$(dirname -- "$SCRIPT_SOURCE")/$LINK_TARGET"
    fi
done
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)"

if [[ "${ALFRED_UPDATE_DRY_RUN:-false}" == "true" ]]; then
    printf '%s\n' "$SCRIPT_DIR/install.sh"
    exit 0
fi

exec "$SCRIPT_DIR/install.sh" --update
