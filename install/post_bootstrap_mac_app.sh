#!/bin/bash
# After bootstrap: symlink Magnitu.app from the repo into ~/Applications and
# put a second symlink on the Desktop. macOS only; only if install is at
# the canonical path ~/Applications/magnitu3.
set -e

INSTALL_DIR="${1:-}"
if [ -z "$INSTALL_DIR" ] || [ ! -d "$INSTALL_DIR" ]; then
    echo "  post_bootstrap_mac_app: missing or bad install directory"
    exit 0
fi

if [[ "$OSTYPE" != darwin* ]]; then
    exit 0
fi

INST=$(cd "$INSTALL_DIR" && pwd)
CANON="${HOME}/Applications/magnitu3"
if [ ! -d "$CANON" ]; then
    exit 0
fi
CANON_REAL=$(cd "$CANON" && pwd)
if [ "$INST" != "$CANON_REAL" ]; then
    echo ""
    echo "  (Skipping Magnitu.app shortcuts: this install is not at $CANON.)"
    echo "        For Desktop / Applications links, use that path and re-run:  bash install/post_bootstrap_mac_app.sh"
    exit 0
fi

APP_SRC="$INST/install/Magnitu.app"
APP_DST="${HOME}/Applications/Magnitu.app"
DESK_DST="${HOME}/Desktop/Magnitu.app"

if [ ! -d "$APP_SRC" ]; then
    echo "  (Skipping Magnitu.app: $APP_SRC not found.)"
    exit 0
fi

mkdir -p "$HOME/Applications" 2>/dev/null || true
mkdir -p "$HOME/Desktop" 2>/dev/null || true

# Replace existing file or link at destination
if [ -e "$APP_DST" ] || [ -L "$APP_DST" ]; then
    rm -rf "$APP_DST"
fi
if [ -e "$DESK_DST" ] || [ -L "$DESK_DST" ]; then
    rm -rf "$DESK_DST"
fi

ln -s "$APP_SRC" "$APP_DST"
ln -s "$APP_DST" "$DESK_DST"

echo ""
echo "  macOS: Magnitu.app is available in Applications and on the Desktop (symlinks to this repo)."
echo ""
