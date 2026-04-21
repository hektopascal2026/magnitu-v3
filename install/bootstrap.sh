#!/bin/bash
# ─────────────────────────────────────────────
#  Magnitu — Remote installer
#
#  For PUBLIC repos:
#    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/hektopascal2026/magnitu-v3/main/install/bootstrap.sh)"
#
#  For PRIVATE repos (one-liner):
#    git clone https://github.com/hektopascal2026/magnitu-v3.git ~/magnitu-v3 && bash ~/magnitu-v3/install/bootstrap.sh
# ─────────────────────────────────────────────

set -e

DEFAULT_URL="https://www.hektopascal.org/seismo/index.php"

# Determine install dir: if we're already inside the repo, use that
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
if [ -f "$SCRIPT_DIR/../main.py" ]; then
    INSTALL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    INSTALL_DIR="$HOME/magnitu"
fi

clear 2>/dev/null || true
echo ""
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Magnitu Installer"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Check prerequisites ──
echo "  [1/5] Checking prerequisites..."

# On macOS, ensure Xcode Command Line Tools are installed (provides git + python3)
if [[ "$OSTYPE" == darwin* ]]; then
    if ! xcode-select -p &>/dev/null; then
        echo ""
        echo "  macOS needs Command Line Tools (provides Python & git)."
        echo "  Installing now — a dialog may appear, click 'Install'."
        echo ""
        xcode-select --install 2>/dev/null
        echo "  After installation finishes, run this command again."
        exit 0
    fi
fi

# Python
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo ""
    echo "  ERROR: Python 3 is required."
    echo "  Install it from https://www.python.org/downloads/"
    echo ""
    exit 1
fi
echo "         Python: $($PY --version 2>&1)"

# Git
if ! command -v git &>/dev/null; then
    echo ""
    echo "  ERROR: git is required."
    echo "  On macOS, run: xcode-select --install"
    echo ""
    exit 1
fi
echo "         git: $(git --version 2>&1 | head -1)"
echo ""

# ── Clone or update ──
echo "  [2/5] Getting Magnitu..."
if [ -f "$INSTALL_DIR/main.py" ]; then
    echo "         Found existing install at $INSTALL_DIR"
    if [ -d "$INSTALL_DIR/.git" ]; then
        cd "$INSTALL_DIR"
        git pull -q origin main 2>/dev/null || true
        echo "         Updated."
    fi
else
    REPO="https://github.com/hektopascal2026/magnitu-v3.git"
    if [ -d "$INSTALL_DIR" ]; then
        echo "         $INSTALL_DIR exists but is not a Magnitu install."
        echo "         Please remove it first: rm -rf $INSTALL_DIR"
        exit 1
    fi
    echo "         Cloning repository..."
    git clone -q "$REPO" "$INSTALL_DIR"
    echo "         Cloned to $INSTALL_DIR"
fi
cd "$INSTALL_DIR"
echo ""

# ── Python environment ──
echo "  [3/5] Setting up Python environment..."
if [ ! -d "$INSTALL_DIR/.venv" ]; then
    $PY -m venv "$INSTALL_DIR/.venv"
fi
"$INSTALL_DIR/.venv/bin/python" -m ensurepip --upgrade -q 2>/dev/null || true
"$INSTALL_DIR/.venv/bin/python" -m pip install -q --upgrade pip 2>/dev/null || true
"$INSTALL_DIR/.venv/bin/python" -m pip install -q -r "$INSTALL_DIR/requirements.txt"
echo "         Done."
echo ""

# ── Configure ──
echo "  [4/5] Configuration"
echo ""

CONFIG_FILE="$INSTALL_DIR/magnitu_config.json"
SKIP_CONFIG=""

# Check if already configured
if [ -f "$CONFIG_FILE" ]; then
    echo "         Existing config found."
    read -r -p "         Reconfigure? (y/N): " RECONFIG
    if [ "$RECONFIG" != "y" ] && [ "$RECONFIG" != "Y" ]; then
        echo "         Keeping existing config."
        SKIP_CONFIG=1
    fi
fi

if [ -z "$SKIP_CONFIG" ]; then
    # Seismo URL
    echo ""
    echo "         Seismo URL (press Enter for default):"
    echo "         Default: $DEFAULT_URL"
    read -r -p "         URL: " SEISMO_URL
    SEISMO_URL="${SEISMO_URL:-$DEFAULT_URL}"
    echo ""

    # API Key
    echo "         Enter your Magnitu API key."
    echo "         (Find it in Seismo → Settings → Magnitu)"
    read -r -p "         API Key: " API_KEY
    while [ -z "$API_KEY" ]; do
        echo "         API key is required."
        read -r -p "         API Key: " API_KEY
    done
    echo ""

    # Write config
    cat > "$CONFIG_FILE" << CONF
{
  "seismo_url": "$SEISMO_URL",
  "api_key": "$API_KEY",
  "min_labels_to_train": 20,
  "recipe_top_keywords": 200,
  "auto_train_after_n_labels": 10,
  "alert_threshold": 0.75,
  "model_architecture": "transformer",
  "transformer_model_name": "xlm-roberta-base",
  "embedding_dim": 768,
  "discovery_lead_blend": 0.0
}
CONF
    echo "         Config saved."
fi

# Reset database if it exists (from a previous install or copy)
DB_FILE="$INSTALL_DIR/magnitu.db"
if [ -f "$DB_FILE" ]; then
    echo ""
    read -r -p "         Reset database for fresh start? (y/N): " RESET_DB
    if [ "$RESET_DB" = "y" ] || [ "$RESET_DB" = "Y" ]; then
        rm -f "$DB_FILE" "$DB_FILE-shm" "$DB_FILE-wal"
        echo "         Database reset."
    fi
fi
echo ""

# ── Test connection ──
echo "  [5/6] Testing connection to Seismo..."
TEST_RESULT=$("$INSTALL_DIR/.venv/bin/python" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
import sync
ok, msg, _ = sync.test_connection()
print(msg)
if not ok:
    sys.exit(1)
" 2>&1) || true
echo "         $TEST_RESULT"
echo ""

# ── Model setup ──
echo "  [6/6] Model setup"
echo ""

# Check if a model profile already exists
HAS_PROFILE=$("$INSTALL_DIR/.venv/bin/python" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
import model_manager
print('yes' if model_manager.has_profile() else 'no')
" 2>&1) || HAS_PROFILE="no"

if [ "$HAS_PROFILE" = "yes" ]; then
    echo "         Model profile already configured."
else
    echo "         Every Magnitu instance needs a model profile."
    echo "         You can create a new model or load an existing .magnitu file."
    echo ""
    echo "         1) Create a new model"
    echo "         2) Load a .magnitu file"
    echo "         3) Skip (set up later in the browser)"
    read -r -p "         Choice [1/2/3]: " MODEL_CHOICE

    if [ "$MODEL_CHOICE" = "1" ]; then
        echo ""
        read -r -p "         Model name (e.g. pascal1): " MODEL_NAME
        while [ -z "$MODEL_NAME" ]; do
            echo "         Name is required."
            read -r -p "         Model name: " MODEL_NAME
        done
        read -r -p "         Short description: " MODEL_DESC
        "$INSTALL_DIR/.venv/bin/python" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
import model_manager
p = model_manager.create_profile('$MODEL_NAME', '$MODEL_DESC')
print('Created model: ' + p['model_name'] + ' (' + p['model_uuid'][:8] + '...)')
" 2>&1
    elif [ "$MODEL_CHOICE" = "2" ]; then
        echo ""
        read -r -p "         Path to .magnitu file: " MAGNITU_FILE
        if [ -f "$MAGNITU_FILE" ]; then
            "$INSTALL_DIR/.venv/bin/python" -c "
import sys
sys.path.insert(0, '$INSTALL_DIR')
import model_manager
result = model_manager.import_model('$MAGNITU_FILE')
print(result.get('message', 'Imported.'))
" 2>&1
        else
            echo "         File not found. You can set up a model later in the browser."
        fi
    else
        echo "         Skipped. You'll be prompted when you first open Magnitu."
    fi
fi
echo ""

# ── Done ──
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Setup complete!"
echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  To start Magnitu, paste this into Terminal:"
echo ""
echo "    $INSTALL_DIR/start.sh"
echo ""
echo "  It will open automatically in your browser."
echo ""
