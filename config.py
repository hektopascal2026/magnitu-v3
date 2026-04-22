"""
Magnitu configuration.
Loaded from environment variables or magnitu_config.json.
"""
import os
import json
from pathlib import Path

VERSION = "3.0.0"

BASE_DIR = Path(__file__).parent

DATA_DIR = Path(os.getenv("MAGNITU_DATA_DIR", str(BASE_DIR)))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = DATA_DIR / "magnitu_config.json"
DB_PATH = DATA_DIR / "magnitu.db"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Defaults
DEFAULTS = {
    "seismo_url": "http://localhost/seismo_0.5/index.php",
    "api_key": "",
    "min_labels_to_train": 20,
    "recipe_top_keywords": 200,
    "auto_train_after_n_labels": 10,
    "alert_threshold": 0.75,
    # Transformer settings (cached embeddings + classifier head)
    "model_architecture": "transformer",     # "tfidf" or "transformer"
    "transformer_model_name": "xlm-roberta-base",
    "embedding_dim": 768,
    "use_gpu": False,  # Use CUDA/MPS when available for embeddings
    # Blend toward investigation_lead in pushed relevance_score (0 = off, max 0.25).
    "discovery_lead_blend": 0.0,
    # Training knobs (all default to no-op):
    # Half-life in days for a time-decay on label age. 0 disables decay.
    "label_time_decay_days": 0,
    # Minimum floor for the decay weight (so very old labels never vanish). 0-1.
    "label_time_decay_floor": 0.2,
    # Multiplier applied to labels that have a non-empty reasoning note. 1 = off.
    "reasoning_weight_boost": 1.0,
    # Regex patterns (or plain phrases) that flag legislative signal in an entry.
    # When matched, the phrase is prepended to the embedded text AND the recipe
    # keyword for it is boosted toward investigation_lead.
    "legal_signal_patterns": [],
    # Which profile is the labeling workspace (slug in URLs for Label/Gemini/etc.).
    # None = use DB default profile until user picks one in Settings.
    "active_profile_id": None,
    # Gemini synthetic labeling settings
    "gemini_api_key": "",
    "gemini_model": "models/gemini-2.5-flash",
}

# Heuristic for Settings UI: suggested recipe_top_keywords by label count (linear 200→400).
RECIPE_TOP_KEYWORDS_SUGGEST_MIN = 200
RECIPE_TOP_KEYWORDS_SUGGEST_MAX = 400
# At this many labels (or more), the suggestion reaches RECIPE_TOP_KEYWORDS_SUGGEST_MAX.
RECIPE_TOP_KEYWORDS_LABELS_FOR_MAX = 2000

# Stored per profile in ``profiles.training_settings`` (JSON). Merged over global
# ``magnitu_config.json`` via ``db.get_effective_config(profile_id)``.
# Legal-signal patterns stay global (they are baked into shared embeddings).
PROFILE_TRAINING_SETTINGS_KEYS = frozenset({
    "min_labels_to_train",
    "recipe_top_keywords",
    "auto_train_after_n_labels",
    "alert_threshold",
    "discovery_lead_blend",
    "label_time_decay_days",
    "label_time_decay_floor",
    "reasoning_weight_boost",
})


def suggested_recipe_top_keywords(label_count: int) -> int:
    """
    Heuristic suggestion for recipe_top_keywords (how many terms the distiller
    exports to the Seismo recipe). Scales linearly from RECIPE_TOP_KEYWORDS_SUGGEST_MIN
    at zero labels to RECIPE_TOP_KEYWORDS_SUGGEST_MAX at RECIPE_TOP_KEYWORDS_LABELS_FOR_MAX
    labels, then stays capped. Used only for UI hints.
    """
    n = max(0, int(label_count))
    t = min(1.0, float(n) / float(RECIPE_TOP_KEYWORDS_LABELS_FOR_MAX))
    return int(
        round(
            RECIPE_TOP_KEYWORDS_SUGGEST_MIN
            + (RECIPE_TOP_KEYWORDS_SUGGEST_MAX - RECIPE_TOP_KEYWORDS_SUGGEST_MIN) * t
        )
    )


def load_config() -> dict:
    """Load config from JSON file, merging with defaults."""
    config = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
            config.update(saved)
        except (json.JSONDecodeError, IOError):
            pass
    return config


def save_config(config: dict):
    """Save config to JSON file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_config() -> dict:
    """Get current config. Re-reads from disk on every call."""
    return load_config()
