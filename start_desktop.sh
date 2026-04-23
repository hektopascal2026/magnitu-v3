#!/bin/bash
# ─────────────────────────────────────────────
#  Magnitu — Start in a native desktop window (pywebview)
#  Usage: ./start_desktop.sh
#
#  One-time: pip install -r requirements-desktop.txt
#  (Or this script will offer to install pywebview if missing.)
# ─────────────────────────────────────────────

# Apple Silicon + Rosetta: re-exec natively so .venv matches arm64 wheels.
if [ "$(uname -m)" = "arm64" ]; then
    _translated=$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)
    if [ "$_translated" = "1" ]; then
        exec arch -arm64 /bin/bash "$0" "$@"
    fi
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$DIR/.venv/bin/python"
PY="$VENV_PY"
export PY
export MAGNITU_VENV_PY="$VENV_PY"
# shellcheck source=install/macos_venv_invoker.sh
. "$DIR/install/macos_venv_invoker.sh"
REQ="$DIR/requirements-desktop.txt"

cd "$DIR" || exit 1

if [ ! -f "$VENV_PY" ]; then
    echo "  No .venv found. Run install/bootstrap.sh or start.sh once to create the environment."
    exit 1
fi

CONFIG_FILE=$(magnitu_venv -c "import os,sys; os.environ.pop('MAGNITU_TEST',None); sys.path.insert(0,r'''$DIR'''); import config; print(config.CONFIG_PATH)" 2>/dev/null) || CONFIG_FILE=""
if [ -z "$CONFIG_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
    echo "  No magnitu_config.json found (expected under your Magnitu data directory). Run install/bootstrap.sh or start.sh once for first-time setup."
    exit 1
fi

if ! magnitu_venv -c "import webview" 2>/dev/null; then
    echo "  Installing desktop dependencies (pywebview)..."
    if ! magnitu_venv -m pip install -q -r "$REQ"; then
        echo "  ERROR: pip install -r $REQ failed."
        exit 1
    fi
    echo "  Done."
    echo ""
fi

# shellcheck source=install/macos_repair_venv_arch.sh
. "$DIR/install/macos_repair_venv_arch.sh"

if [ "$(uname -m)" = "arm64" ]; then
    exec arch -arm64 "$VENV_PY" "$DIR/desktop.py"
else
    exec "$VENV_PY" "$DIR/desktop.py"
fi
