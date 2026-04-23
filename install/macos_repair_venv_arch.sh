# Repair mixed arm64 / x86_64 wheels in a venv (e.g. pydantic_core built for
# the wrong slice). On macOS only; no-op elsewhere.
# Expect:  PY  — path to the venv's python executable (e.g. .../.venv/bin/python)
# Sourced from mac_open_desktop.sh, start.sh, start_desktop.sh.

if [[ "$OSTYPE" != darwin* ]]; then
    return 0 2>/dev/null || true
fi
[ -n "${PY:-}" ] && [ -x "$PY" ] || return 0 2>/dev/null || true

REPAIR_ERR="$(mktemp -t magnitu-venv-repair.XXXXXX)"
if ! "$PY" -c "import pydantic_core" 2>"$REPAIR_ERR"; then
    if grep -q "incompatible architecture" "$REPAIR_ERR" 2>/dev/null; then
        echo "  Repairing: a native package (e.g. pydantic_core) does not match this Python — reinstalling for the current architecture..."
        if ! "$PY" -m pip install -q --no-cache-dir --force-reinstall "pydantic" "pydantic-core" 2>&1; then
            echo "  (Automatic fix failed. Remove the .venv folder, then in Terminal: arch -arm64 bash install/bootstrap.sh )"
        fi
    fi
fi
rm -f "$REPAIR_ERR"
return 0 2>/dev/null || true
