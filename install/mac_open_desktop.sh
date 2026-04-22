#!/bin/bash
# Launch Magnitu in the desktop (pywebview) window after git sync and deps.
# Intended for: double-click Magnitu.app, or: bash install/mac_open_desktop.sh
# Canonical install: $HOME/Applications/magnitu3
set -e

REPO="${MAGNITU_HOME:-$HOME/Applications/magnitu3}"
DIR="$REPO"
PY="$REPO/.venv/bin/python"
STAMP="$REPO/.venv/.deps_ok"
STAMP_DESKTOP="$REPO/.venv/.deps_desktop_ok"
REQ="$REPO/requirements.txt"
REQD="$REPO/requirements-desktop.txt"

msg_alert() {
    local title="${1:-Magnitu}"
    local body="$2"
    if command -v osascript &>/dev/null; then
        osascript -e "display alert \"$title\" message \"$body\"" 2>/dev/null || true
    else
        echo "  $body" >&2
    fi
}

if [ ! -d "$REPO" ] || [ ! -f "$REPO/main.py" ]; then
    msg_alert "Magnitu" "Clone the repo to ~/Applications/magnitu3, then run bash install/bootstrap.sh in Terminal."
    exit 1
fi
cd "$REPO"

if [ ! -f "$PY" ] || [ ! -f "$REPO/magnitu_config.json" ]; then
    msg_alert "Magnitu" "Run the installer first: bash install/bootstrap.sh (in your magnitu3 folder, in Terminal)."
    exit 1
fi

# shellcheck source=mag_git_sync.sh
. "$REPO/install/mag_git_sync.sh"

DEPS_CHANGED=0
if ! mag_sync_github; then
    msg_alert "Magnitu" "Git update failed. In Terminal: cd ~/Applications/magnitu3 and git status"
    exit 1
fi
if [ "$MAG_GIT_UPDATED" = "1" ]; then
    DEPS_CHANGED=1
fi

NEED_CHECK=0
if [ "$DEPS_CHANGED" = "1" ]; then
    NEED_CHECK=1
elif [ ! -f "$STAMP" ]; then
    NEED_CHECK=1
elif [ -f "$REQ" ] && [ "$REQ" -nt "$STAMP" ]; then
    NEED_CHECK=1
fi

if [ "$NEED_CHECK" = "1" ]; then
    MISSING=$("$PY" -c "
missing = []
for mod in ['uvicorn', 'fastapi', 'httpx', 'sklearn', 'torch', 'transformers']:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
print(','.join(missing))
" 2>/dev/null) || true

    if [ -n "$MISSING" ] || [ "$DEPS_CHANGED" = "1" ]; then
        if ! "$PY" -m pip install -q -r "$REQ" 2>&1; then
            msg_alert "Magnitu" "pip install failed. Check network, then in Terminal: pip install -r requirements.txt"
            exit 1
        fi
    fi
    touch "$STAMP"
fi

# Desktop extra (pywebview) — re-check if requirements-desktop.txt changed
NEED_DESKTOP=0
if [ ! -f "$STAMP_DESKTOP" ]; then
    NEED_DESKTOP=1
elif [ -f "$REQD" ] && [ "$REQD" -nt "$STAMP_DESKTOP" ]; then
    NEED_DESKTOP=1
fi
if [ "$NEED_DESKTOP" = "1" ] || ! "$PY" -c "import webview" 2>/dev/null; then
    if ! "$PY" -m pip install -q -r "$REQD" 2>&1; then
        msg_alert "Magnitu" "Could not install requirements-desktop.txt. Run pip install -r requirements-desktop.txt in Terminal."
        exit 1
    fi
    touch "$STAMP_DESKTOP"
fi

exec "$PY" "$REPO/desktop.py"
