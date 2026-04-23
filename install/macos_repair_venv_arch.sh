# Repair mixed arm64 / x86_64 native wheels in a venv. On macOS only; no-op elsewhere.
# If any extension module (e.g. pydantic_core, numpy) is the wrong arch for this
# Python, reinstall the full tree from requirements.txt so torch/sklearn match too.
# Expect:  PY  — venv python; REPO or DIR set (path to magnitu3 clone). Uses magnitu_venv
# to force arch -arm64 on Apple Silicon. Sourced from mac_open, start, start_desktop.

if [[ "$OSTYPE" != darwin* ]]; then
    return 0 2>/dev/null || true
fi
[ -n "${PY:-}" ] && [ -x "$PY" ] || return 0 2>/dev/null || true

_INV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0})" 2>/dev/null && pwd)" || _INV_DIR=""
# shellcheck source=./macos_venv_invoker.sh
[ -n "$_INV_DIR" ] && [ -f "$_INV_DIR/macos_venv_invoker.sh" ] && . "$_INV_DIR/macos_venv_invoker.sh"
[ "$(type -t magnitu_venv 2>/dev/null)" = function ] || return 0 2>/dev/null || true

MAGNITU_VENV_PY="${MAGNITU_VENV_PY:-$PY}"
export MAGNITU_VENV_PY

ROOT="${REPO:-$DIR}"
REQ=""
if [ -n "$ROOT" ] && [ -f "$ROOT/requirements.txt" ]; then
    REQ="$ROOT/requirements.txt"
fi

REPAIR_ERR="$(mktemp -t magnitu-venv-repair.XXXXXX)"
set +e
# Pydantic alone can "repair" while numpy/torch still point at the wrong CPU slice.
magnitu_venv -c "
import importlib, sys
for name in ('pydantic_core', 'numpy'):
    try:
        importlib.import_module(name)
    except Exception as e:
        if 'incompatible architecture' in str(e):
            sys.exit(3)
        raise
" 2>"$REPAIR_ERR"
_rc=$?
set -e
if [ "$_rc" -eq 0 ]; then
    rm -f "$REPAIR_ERR"
    return 0 2>/dev/null || true
fi
if [ "$_rc" -ne 3 ] && ! grep -q "incompatible architecture" "$REPAIR_ERR" 2>/dev/null; then
    rm -f "$REPAIR_ERR"
    return 0 2>/dev/null || true
fi
rm -f "$REPAIR_ERR"

echo "  Repairing: native Python wheels (numpy, torch, etc.) do not match this interpreter — reinstalling for the current architecture (may take a few minutes)..."
if [ -n "$REQ" ]; then
    if ! magnitu_venv -m pip install -q --no-cache-dir --force-reinstall -r "$REQ" 2>&1; then
        echo "  (Automatic fix failed. Remove the .venv folder, then: arch -arm64 bash install/bootstrap.sh )" >&2
    fi
else
    if ! magnitu_venv -m pip install -q --no-cache-dir --force-reinstall "numpy" "pydantic" "pydantic-core" 2>&1; then
        echo "  (Automatic fix failed. Remove the .venv folder, then: arch -arm64 bash install/bootstrap.sh )" >&2
    fi
fi
return 0 2>/dev/null || true
