#!/bin/bash
# After bootstrap: symlink Magnitu.app from the repo into ~/Applications and
# put a second symlink on the Desktop. macOS only; only if install is at
# the canonical path ~/Applications/magnitu3.
set -e

# Argument $1 = repo root (as bootstrap passes). If omitted, infer from this script
# (so:  cd ~/Applications/magnitu3 && bash install/post_bootstrap_mac_app.sh  works).
INSTALL_DIR="${1:-}"
if [ -z "$INSTALL_DIR" ] || [ ! -d "$INSTALL_DIR" ] || [ ! -f "$INSTALL_DIR/main.py" ]; then
    _sdir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0})" 2>/dev/null && pwd)" || _sdir=""
    if [ -n "$_sdir" ] && [ -f "$_sdir/../main.py" ]; then
        INSTALL_DIR="$(cd "$_sdir/.." && pwd)"
    else
        INSTALL_DIR="${HOME}/Applications/magnitu3"
    fi
fi
if [ ! -d "$INSTALL_DIR" ] || [ ! -f "$INSTALL_DIR/main.py" ]; then
    echo "  post_bootstrap_mac_app: missing or bad install directory: $INSTALL_DIR"
    echo "  Usage:  cd ~/Applications/magnitu3 && bash install/post_bootstrap_mac_app.sh"
    echo "     or:  bash install/post_bootstrap_mac_app.sh  /path/to/magnitu-v3"
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
    echo "        For Desktop / Applications links, move the clone to $CANON then run:"
    echo "          cd $CANON && bash install/post_bootstrap_mac_app.sh"
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
