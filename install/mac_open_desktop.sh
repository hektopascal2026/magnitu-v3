#!/bin/bash
# Launch Magnitu in the desktop (pywebview) window after git sync and deps.
# Intended for: double-click Magnitu.app, or: bash install/mac_open_desktop.sh
# Canonical install: $HOME/Applications/magnitu3
#
# All output (including errors) is appended to a log; on failure the log is opened
# for easy remote debugging. See LOG_FILE below.
set -e

REPO="${MAGNITU_HOME:-$HOME/Applications/magnitu3}"
INSTALL_LOG_DIR="${HOME}/Library/Logs/Magnitu"
mkdir -p "$INSTALL_LOG_DIR" 2>/dev/null || true
if [ -d "$REPO" ] && [ -f "$REPO/main.py" ]; then
    LOG_FILE="$REPO/.magnitu_desktop_last.log"
else
    LOG_FILE="${INSTALL_LOG_DIR}/magnitu_desktop_last.log"
fi

{
    echo ""
    echo "==== Magnitu desktop $(date '+%Y-%m-%d %H:%M:%S %Z') pid=$$ ===="
} >>"$LOG_FILE"

# Mirror stdout/stderr to log; when run from Terminal, you still see output.
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Set when we already showed an osascript (avoid duplicate "unexpected" dialog).
MUSER_NOTIFIED=0

on_exit() {
    local ec=$?
    if [ "$ec" -eq 0 ]; then
        return 0
    fi
    if [ "$MUSER_NOTIFIED" != "1" ] && command -v osascript &>/dev/null; then
        osascript -e "display alert \"Magnitu\" message \"Something went wrong (exit $ec). A detailed log will open.\""
    fi
    if [ -f "$LOG_FILE" ]; then
        open -e "$LOG_FILE" 2>/dev/null || true
    fi
}

trap on_exit EXIT

msg_alert() {
    local title="${1:-Magnitu}"
    local body="$2"
    echo "[$title] $body"
    echo "  (Full log: $LOG_FILE)"
    MUSER_NOTIFIED=1
    if command -v osascript &>/dev/null; then
        osascript -e "display alert \"$title\" message \"$body\""
    else
        echo "  $body" >&2
    fi
}

if [ ! -d "$REPO" ] || [ ! -f "$REPO/main.py" ]; then
    msg_alert "Magnitu" "Clone the repo to ~/Applications/magnitu3, then run bash install/bootstrap.sh in Terminal."
    exit 1
fi
cd "$REPO"

if [ ! -f "$REPO/.venv/bin/python" ] || [ ! -f "$REPO/magnitu_config.json" ]; then
    msg_alert "Magnitu" "Run the installer first: bash install/bootstrap.sh in your magnitu3 folder in Terminal."
    exit 1
fi

PY="$REPO/.venv/bin/python"
STAMP="$REPO/.venv/.deps_ok"
STAMP_DESKTOP="$REPO/.venv/.deps_desktop_ok"
REQ="$REPO/requirements.txt"
REQD="$REPO/requirements-desktop.txt"

# shellcheck source=mag_git_sync.sh
. "$REPO/install/mag_git_sync.sh"

DEPS_CHANGED=0
if ! mag_sync_github; then
    msg_alert "Magnitu" "Git update failed. In Terminal, run: cd ~/Applications/magnitu3  then  git status"
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
            msg_alert "Magnitu" "pip install failed. Check the log (opening), then in Terminal: pip install -r requirements.txt"
            exit 1
        fi
    fi
    touch "$STAMP"
fi

NEED_DESKTOP=0
if [ ! -f "$STAMP_DESKTOP" ]; then
    NEED_DESKTOP=1
elif [ -f "$REQD" ] && [ "$REQD" -nt "$STAMP_DESKTOP" ]; then
    NEED_DESKTOP=1
fi
if [ "$NEED_DESKTOP" = "1" ] || ! "$PY" -c "import webview" 2>/dev/null; then
    if ! "$PY" -m pip install -q -r "$REQD" 2>&1; then
        msg_alert "Magnitu" "Could not install requirements-desktop.txt. See the log (opening), then: pip install -r requirements-desktop.txt"
        exit 1
    fi
    touch "$STAMP_DESKTOP"
fi

# Do not use exec: retain EXIT trap so Python errors still open the log.
set +e
"$PY" "$REPO/desktop.py"
rc=$?
set -e
exit "$rc"
