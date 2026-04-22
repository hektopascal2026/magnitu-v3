"""
Local SQLite database for Magnitu.
Stores cached entries, user labels, model history, and sync log.

Multi-profile support: labels and models are scoped to a profile_id.
Entries and embeddings are shared across all profiles (computed once).
"""
import re
import sqlite3
import json
from typing import Optional, List
from datetime import datetime
from config import DB_PATH, load_config, save_config


def get_db() -> sqlite3.Connection:
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def slugify(name: str) -> str:
    """Convert a display name to a URL-safe slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "default"


def _migrate_db(conn: sqlite3.Connection):
    """Run all schema migrations. Idempotent."""
    # ── Legacy column additions ──────────────────────────────────────────────
    cursor = conn.execute("PRAGMA table_info(entries)")
    entry_cols = {row[1] for row in cursor.fetchall()}
    if "embedding" not in entry_cols:
        conn.execute("ALTER TABLE entries ADD COLUMN embedding BLOB DEFAULT NULL")

    cursor = conn.execute("PRAGMA table_info(labels)")
    label_cols = {row[1] for row in cursor.fetchall()}
    if "reasoning" not in label_cols:
        conn.execute("ALTER TABLE labels ADD COLUMN reasoning TEXT DEFAULT ''")

    label_cols = {row[1] for row in conn.execute("PRAGMA table_info(labels)").fetchall()}
    if "label_source" not in label_cols:
        conn.execute("ALTER TABLE labels ADD COLUMN label_source TEXT DEFAULT ''")

    cursor = conn.execute("PRAGMA table_info(models)")
    model_cols = {row[1] for row in cursor.fetchall()}
    if "architecture" not in model_cols:
        conn.execute("ALTER TABLE models ADD COLUMN architecture TEXT DEFAULT 'tfidf'")

    # ── Profiles table ───────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            slug        TEXT    NOT NULL UNIQUE,
            display_name TEXT   NOT NULL,
            description TEXT    DEFAULT '',
            seismo_url  TEXT    DEFAULT '',
            api_key     TEXT    DEFAULT '',
            model_uuid  TEXT    DEFAULT '',
            is_default  INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now'))
        )
    """)

    # Migrate legacy model_profile → profiles (runs once when profiles is empty)
    n_profiles = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    if n_profiles == 0:
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if "model_profile" in tables:
                row = conn.execute(
                    "SELECT * FROM model_profile ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    # sqlite3.Row doesn't have .get() — convert to dict first
                    row_dict = dict(row)
                    slug = slugify(row_dict["model_name"])
                    conn.execute("""
                        INSERT INTO profiles
                            (id, slug, display_name, description, model_uuid, is_default)
                        VALUES (1, ?, ?, ?, ?, 1)
                    """, (slug, row_dict["model_name"],
                          row_dict.get("description") or "",
                          row_dict.get("model_uuid") or ""))
                    n_profiles = 1
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning("Profile migration failed: %s", e)

    # ── Add profile_id to models ─────────────────────────────────────────────
    model_cols = {r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()}
    if "profile_id" not in model_cols:
        conn.execute("ALTER TABLE models ADD COLUMN profile_id INTEGER NOT NULL DEFAULT 1")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_profile ON models(profile_id)"
        )

    # ── Recreate labels with profile_id ─────────────────────────────────────
    label_cols = {r[1] for r in conn.execute("PRAGMA table_info(labels)").fetchall()}
    if "profile_id" not in label_cols:
        conn.execute("ALTER TABLE labels RENAME TO _labels_legacy")
        conn.execute("""
            CREATE TABLE labels (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id  INTEGER NOT NULL DEFAULT 1,
                entry_type  TEXT    NOT NULL,
                entry_id    INTEGER NOT NULL,
                label       TEXT    NOT NULL,
                reasoning   TEXT    DEFAULT '',
                created_at  TEXT    DEFAULT (datetime('now')),
                updated_at  TEXT    DEFAULT (datetime('now')),
                UNIQUE(profile_id, entry_type, entry_id)
            )
        """)
        # Be defensive: old DBs may lack created_at/updated_at/reasoning
        legacy_cols = {r[1] for r in conn.execute("PRAGMA table_info(_labels_legacy)").fetchall()}
        sel_created = "created_at" if "created_at" in legacy_cols else "datetime('now')"
        sel_updated = "updated_at" if "updated_at" in legacy_cols else "datetime('now')"
        sel_reasoning = "COALESCE(reasoning, '')" if "reasoning" in legacy_cols else "''"
        conn.execute("""
            INSERT INTO labels
                (profile_id, entry_type, entry_id, label, reasoning, created_at, updated_at)
            SELECT 1, entry_type, entry_id, label,
                   {reasoning}, {created}, {updated}
            FROM _labels_legacy
        """.format(reasoning=sel_reasoning, created=sel_created, updated=sel_updated))
        conn.execute("DROP TABLE _labels_legacy")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_labels_profile ON labels(profile_id)"
        )

    # ── Add profile_id to sync_log (nullable — pull is global) ───────────────
    sl_cols = {r[1] for r in conn.execute("PRAGMA table_info(sync_log)").fetchall()}
    if "profile_id" not in sl_cols:
        conn.execute(
            "ALTER TABLE sync_log ADD COLUMN profile_id INTEGER DEFAULT NULL"
        )

    model_cols2 = {r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()}
    if "label_distribution" not in model_cols2:
        conn.execute(
            "ALTER TABLE models ADD COLUMN label_distribution TEXT DEFAULT '{}'"
        )

    prof_cols = {r[1] for r in conn.execute("PRAGMA table_info(profiles)").fetchall()}
    if "accent_color" not in prof_cols:
        conn.execute("ALTER TABLE profiles ADD COLUMN accent_color TEXT")

    prof_cols = {r[1] for r in conn.execute("PRAGMA table_info(profiles)").fetchall()}
    if "training_settings" not in prof_cols:
        conn.execute(
            "ALTER TABLE profiles ADD COLUMN training_settings TEXT DEFAULT '{}'"
        )

    prof_cols = {r[1] for r in conn.execute("PRAGMA table_info(profiles)").fetchall()}
    if "gemini_persona" not in prof_cols:
        conn.execute(
            "ALTER TABLE profiles ADD COLUMN gemini_persona TEXT DEFAULT NULL"
        )

    conn.commit()


def init_db():
    """Create all tables if they don't exist, then run migrations."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entries (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_type      TEXT    NOT NULL,
            entry_id        INTEGER NOT NULL,
            title           TEXT    DEFAULT '',
            description     TEXT    DEFAULT '',
            content         TEXT    DEFAULT '',
            link            TEXT    DEFAULT '',
            author          TEXT    DEFAULT '',
            published_date  TEXT    DEFAULT '',
            source_name     TEXT    DEFAULT '',
            source_category TEXT    DEFAULT '',
            source_type     TEXT    DEFAULT '',
            embedding       BLOB    DEFAULT NULL,
            fetched_at      TEXT    DEFAULT (datetime('now')),
            UNIQUE(entry_type, entry_id)
        );

        CREATE TABLE IF NOT EXISTS labels (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id  INTEGER NOT NULL DEFAULT 1,
            entry_type  TEXT    NOT NULL,
            entry_id    INTEGER NOT NULL,
            label       TEXT    NOT NULL,
            reasoning   TEXT    DEFAULT '',
            label_source TEXT   DEFAULT '',
            created_at  TEXT    DEFAULT (datetime('now')),
            updated_at  TEXT    DEFAULT (datetime('now')),
            UNIQUE(profile_id, entry_type, entry_id)
        );

        CREATE TABLE IF NOT EXISTS models (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id      INTEGER NOT NULL DEFAULT 1,
            version         INTEGER NOT NULL,
            accuracy        REAL    DEFAULT 0.0,
            f1_score        REAL    DEFAULT 0.0,
            precision_score REAL    DEFAULT 0.0,
            recall_score    REAL    DEFAULT 0.0,
            label_count     INTEGER DEFAULT 0,
            label_distribution TEXT DEFAULT '{}',
            feature_count   INTEGER DEFAULT 0,
            model_path      TEXT    DEFAULT '',
            recipe_path     TEXT    DEFAULT '',
            recipe_quality  REAL    DEFAULT 0.0,
            architecture    TEXT    DEFAULT 'tfidf',
            trained_at      TEXT    DEFAULT (datetime('now')),
            is_active       INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id  INTEGER DEFAULT NULL,
            direction   TEXT    NOT NULL,
            items_count INTEGER DEFAULT 0,
            details     TEXT    DEFAULT '',
            synced_at   TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS model_profile (
            id          INTEGER PRIMARY KEY,
            model_name  TEXT    NOT NULL,
            model_uuid  TEXT    NOT NULL,
            description TEXT    DEFAULT '',
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            slug        TEXT    NOT NULL UNIQUE,
            display_name TEXT   NOT NULL,
            description TEXT    DEFAULT '',
            seismo_url  TEXT    DEFAULT '',
            api_key     TEXT    DEFAULT '',
            model_uuid  TEXT    DEFAULT '',
            accent_color TEXT   DEFAULT NULL,
            is_default  INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now')),
            gemini_persona TEXT DEFAULT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_entries_type_id  ON entries(entry_type, entry_id);
        CREATE INDEX IF NOT EXISTS idx_labels_type_id   ON labels(entry_type, entry_id);
        CREATE INDEX IF NOT EXISTS idx_labels_label     ON labels(label);
        CREATE INDEX IF NOT EXISTS idx_models_active    ON models(is_active);
    """)
    _migrate_db(conn)
    conn.commit()
    conn.close()


# ─── Profile operations ──────────────────────────────────────────────────────

def get_all_profiles() -> List[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM profiles ORDER BY is_default DESC, id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_profile_by_id(profile_id: int) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_profile_by_slug(slug: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM profiles WHERE slug = ?", (slug,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _normalize_training_settings_dict(d: dict) -> None:
    """Clamp per-profile training overrides to valid ranges (mutates in place)."""
    if "discovery_lead_blend" in d:
        try:
            b = float(d.get("discovery_lead_blend", 0.0) or 0.0)
        except (TypeError, ValueError):
            b = 0.0
        d["discovery_lead_blend"] = max(0.0, min(0.25, b))
    if "label_time_decay_days" in d:
        try:
            hl = int(float(d.get("label_time_decay_days", 0) or 0))
        except (TypeError, ValueError):
            hl = 0
        d["label_time_decay_days"] = max(0, min(3650, hl))
    if "label_time_decay_floor" in d:
        try:
            fl = float(d.get("label_time_decay_floor", 0.2) or 0.0)
        except (TypeError, ValueError):
            fl = 0.2
        d["label_time_decay_floor"] = max(0.0, min(1.0, fl))
    if "reasoning_weight_boost" in d:
        try:
            rb = float(d.get("reasoning_weight_boost", 1.0) or 1.0)
        except (TypeError, ValueError):
            rb = 1.0
        d["reasoning_weight_boost"] = max(0.0, min(5.0, rb))
    if "min_labels_to_train" in d:
        try:
            d["min_labels_to_train"] = max(5, min(500, int(d["min_labels_to_train"])))
        except (TypeError, ValueError):
            d["min_labels_to_train"] = 20
    if "recipe_top_keywords" in d:
        try:
            d["recipe_top_keywords"] = max(50, min(1000, int(d["recipe_top_keywords"])))
        except (TypeError, ValueError):
            d["recipe_top_keywords"] = 200
    if "auto_train_after_n_labels" in d:
        try:
            d["auto_train_after_n_labels"] = max(1, int(d["auto_train_after_n_labels"]))
        except (TypeError, ValueError):
            d["auto_train_after_n_labels"] = 10
    if "alert_threshold" in d:
        try:
            at = float(d.get("alert_threshold", 0.75) or 0.75)
        except (TypeError, ValueError):
            at = 0.75
        d["alert_threshold"] = max(0.0, min(1.0, at))
    if "gemini_mode" in d:
        m = str(d.get("gemini_mode", "single")).lower().strip()
        d["gemini_mode"] = m if m in ("single", "batch") else "single"


def get_profile_training_settings(profile_id: int) -> dict:
    """Per-profile training overrides (merged over global config)."""
    conn = get_db()
    row = conn.execute(
        "SELECT training_settings FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    conn.close()
    if not row or row[0] is None or str(row[0]).strip() == "":
        return {}
    try:
        raw = json.loads(row[0])
        return raw if isinstance(raw, dict) else {}
    except json.JSONDecodeError:
        return {}


def merge_profile_training_settings(profile_id: int, partial: dict) -> None:
    """Merge ``partial`` into stored training_settings and persist."""
    current = get_profile_training_settings(profile_id)
    current.update(partial)
    _normalize_training_settings_dict(current)
    conn = get_db()
    conn.execute(
        "UPDATE profiles SET training_settings = ? WHERE id = ?",
        (json.dumps(current, ensure_ascii=False), profile_id)
    )
    conn.commit()
    conn.close()


def get_effective_config(profile_id: int) -> dict:
    """Global magnitu_config merged with this profile's training_settings."""
    from config import get_config, PROFILE_TRAINING_SETTINGS_KEYS
    base = get_config()
    overrides = get_profile_training_settings(profile_id)
    out = dict(base)
    for k in PROFILE_TRAINING_SETTINGS_KEYS:
        if k in overrides:
            out[k] = overrides[k]
    return out


def get_default_profile() -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM profiles ORDER BY is_default DESC, id ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def has_any_profile() -> bool:
    conn = get_db()
    n = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    conn.close()
    return n > 0


def create_profile(slug: str, display_name: str, description: str = "",
                   seismo_url: str = "", api_key: str = "") -> dict:
    """Create a new profile. Returns the created profile dict."""
    import uuid
    conn = get_db()
    model_uuid = uuid.uuid4().hex
    # First profile becomes the default
    n = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    is_default = 1 if n == 0 else 0
    conn.execute("""
        INSERT INTO profiles (slug, display_name, description, seismo_url, api_key,
                              model_uuid, is_default)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (slug, display_name, description, seismo_url, api_key, model_uuid, is_default))
    profile_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    conn.close()
    return dict(row)


def set_profile_accent_color(profile_id: int, accent_hex: str) -> None:
    """Persist validated #rrggbb from Seismo magnitu_status (best-effort)."""
    conn = get_db()
    conn.execute(
        "UPDATE profiles SET accent_color=? WHERE id=?", (accent_hex, profile_id)
    )
    conn.commit()
    conn.close()


def get_profile_gemini_persona(profile_id: int) -> Optional[str]:
    conn = get_db()
    row = conn.execute(
        "SELECT gemini_persona FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    conn.close()
    if not row or row[0] is None:
        return None
    return str(row[0])


def set_profile_gemini_persona(profile_id: int, persona: Optional[str]) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE profiles SET gemini_persona = ? WHERE id = ?", (persona, profile_id)
    )
    conn.commit()
    conn.close()


def update_profile(profile_id: int, display_name: Optional[str] = None,
                   description: Optional[str] = None,
                   seismo_url: Optional[str] = None,
                   api_key: Optional[str] = None,
                   is_default: Optional[int] = None) -> None:
    conn = get_db()
    if is_default == 1:
        conn.execute("UPDATE profiles SET is_default = 0")
    fields = []
    vals: list = []
    for col, val in [("display_name", display_name), ("description", description),
                     ("seismo_url", seismo_url), ("api_key", api_key),
                     ("is_default", is_default)]:
        if val is not None:
            fields.append(f"{col} = ?")
            vals.append(val)
    if fields:
        vals.append(profile_id)
        conn.execute(
            "UPDATE profiles SET {} WHERE id = ?".format(", ".join(fields)), vals
        )
    conn.commit()
    conn.close()


def delete_profile(profile_id: int) -> None:
    """Delete a profile and all its labels and model records."""
    conn = get_db()
    conn.execute("DELETE FROM labels WHERE profile_id = ?", (profile_id,))
    conn.execute("DELETE FROM models WHERE profile_id = ?", (profile_id,))
    conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
    conn.commit()
    conn.close()


def set_profile_as_default(profile_id: int) -> None:
    conn = get_db()
    conn.execute("UPDATE profiles SET is_default = 0")
    conn.execute("UPDATE profiles SET is_default = 1 WHERE id = ?", (profile_id,))
    conn.commit()
    conn.close()


def get_active_profile() -> Optional[dict]:
    """Workspace profile for Label / Gemini / Push (persisted in magnitu_config.json)."""
    cfg = load_config()
    raw = cfg.get("active_profile_id")
    if raw is not None:
        try:
            pid = int(raw)
        except (TypeError, ValueError):
            pid = None
        if pid is not None:
            row = get_profile_by_id(pid)
            if row:
                return row
    return get_default_profile()


def set_active_profile_id(profile_id: int) -> None:
    """Persist which profile is active for the labeling workspace."""
    row = get_profile_by_id(profile_id)
    if not row:
        raise ValueError("Profile not found")
    cfg = load_config()
    cfg["active_profile_id"] = profile_id
    save_config(cfg)


def clear_active_profile_if_deleted(deleted_profile_id: int) -> None:
    """After deleting a profile, unset active_profile_id if it pointed there."""
    cfg = load_config()
    try:
        cur = int(cfg.get("active_profile_id"))
    except (TypeError, ValueError):
        return
    if cur == deleted_profile_id:
        cfg["active_profile_id"] = None
        save_config(cfg)


# ─── Entry operations ────────────────────────────────────────────────────────

def upsert_entry(entry: dict):
    """Insert or update a cached entry from seismo.
    Invalidates cached embedding when text content changes."""
    conn = get_db()
    conn.execute("""
        INSERT INTO entries (entry_type, entry_id, title, description, content, link, author,
                            published_date, source_name, source_category, source_type)
        VALUES (:entry_type, :entry_id, :title, :description, :content, :link, :author,
                :published_date, :source_name, :source_category, :source_type)
        ON CONFLICT(entry_type, entry_id) DO UPDATE SET
            title=excluded.title, description=excluded.description, content=excluded.content,
            link=excluded.link, author=excluded.author, published_date=excluded.published_date,
            source_name=excluded.source_name, source_category=excluded.source_category,
            source_type=excluded.source_type, fetched_at=datetime('now'),
            embedding = CASE
                WHEN entries.title != excluded.title
                  OR entries.description != excluded.description
                  OR entries.content != excluded.content
                  OR IFNULL(entries.source_name, '') != IFNULL(excluded.source_name, '')
                  OR IFNULL(entries.source_category, '') != IFNULL(excluded.source_category, '')
                  OR IFNULL(entries.source_type, '') != IFNULL(excluded.source_type, '')
                THEN NULL
                ELSE entries.embedding
            END
    """, entry)
    conn.commit()
    conn.close()


def upsert_entries(entries: List[dict]):
    """Batch upsert entries.
    Invalidates cached embedding when text content changes."""
    conn = get_db()
    conn.executemany("""
        INSERT INTO entries (entry_type, entry_id, title, description, content, link, author,
                            published_date, source_name, source_category, source_type)
        VALUES (:entry_type, :entry_id, :title, :description, :content, :link, :author,
                :published_date, :source_name, :source_category, :source_type)
        ON CONFLICT(entry_type, entry_id) DO UPDATE SET
            title=excluded.title, description=excluded.description, content=excluded.content,
            link=excluded.link, author=excluded.author, published_date=excluded.published_date,
            source_name=excluded.source_name, source_category=excluded.source_category,
            source_type=excluded.source_type, fetched_at=datetime('now'),
            embedding = CASE
                WHEN entries.title != excluded.title
                  OR entries.description != excluded.description
                  OR entries.content != excluded.content
                  OR IFNULL(entries.source_name, '') != IFNULL(excluded.source_name, '')
                  OR IFNULL(entries.source_category, '') != IFNULL(excluded.source_category, '')
                  OR IFNULL(entries.source_type, '') != IFNULL(excluded.source_type, '')
                THEN NULL
                ELSE entries.embedding
            END
    """, entries)
    conn.commit()
    conn.close()


def get_unlabeled_entries(limit: int = 30, entry_type: Optional[str] = None,
                          profile_id: int = 1) -> List[dict]:
    """Get entries not yet labeled in this profile, newest first."""
    conn = get_db()
    sql = """
        SELECT e.* FROM entries e
        LEFT JOIN labels l ON e.entry_type = l.entry_type
                          AND e.entry_id   = l.entry_id
                          AND l.profile_id = ?
        WHERE l.id IS NULL
    """
    params: list = [profile_id]
    if entry_type:
        sql += " AND e.entry_type = ?"
        params.append(entry_type)
    sql += " ORDER BY e.published_date DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    # One row per (entry_type, entry_id); tolerate type drift (int vs str id) from legacy/sync.
    out = []
    seen = set()
    for r in rows:
        d = dict(r)
        try:
            eid = int(d["entry_id"])
        except (TypeError, ValueError, KeyError):
            eid = d.get("entry_id")
        k = (str(d.get("entry_type") or ""), eid)
        if k not in seen:
            seen.add(k)
            d["entry_id"] = eid
            out.append(d)
    return out


def get_all_entries() -> List[dict]:
    """Get all cached entries (global, not profile-scoped)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM entries ORDER BY published_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_entries(days: int = 7) -> List[dict]:
    """Get entries published within the last N days."""
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM entries
        WHERE published_date >= date('now', ?)
        ORDER BY published_date DESC
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_labeled_entries(profile_id: int = 1) -> List[dict]:
    """Get all entries that have a user label in this profile."""
    conn = get_db()
    rows = conn.execute("""
        SELECT e.*, l.label as user_label
        FROM entries e
        JOIN labels l ON e.entry_type = l.entry_type
                     AND e.entry_id   = l.entry_id
                     AND l.profile_id = ?
        ORDER BY e.published_date DESC
    """, (profile_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_entry_count() -> int:
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    conn.close()
    return count


# ─── Label operations ────────────────────────────────────────────────────────

def set_label(entry_type: str, entry_id: int, label: str,
              reasoning: str = "", profile_id: int = 1,
              label_source: str = ""):
    """Set or update a label for an entry within a profile.

    label_source: empty for manual labels; \"Gemini\" for synthetic (kept local; not sent to Seismo).
    """
    conn = get_db()
    conn.execute("""
        INSERT INTO labels (profile_id, entry_type, entry_id, label, reasoning, label_source)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, entry_type, entry_id) DO UPDATE SET
            label=excluded.label, reasoning=excluded.reasoning,
            label_source=excluded.label_source, updated_at=datetime('now')
    """, (profile_id, entry_type, entry_id, label, reasoning, label_source or ""))
    conn.commit()
    conn.close()


def remove_label(entry_type: str, entry_id: int, profile_id: int = 1):
    conn = get_db()
    conn.execute(
        "DELETE FROM labels WHERE profile_id = ? AND entry_type = ? AND entry_id = ?",
        (profile_id, entry_type, entry_id)
    )
    conn.commit()
    conn.close()


def get_label(entry_type: str, entry_id: int, profile_id: int = 1) -> Optional[str]:
    conn = get_db()
    row = conn.execute(
        "SELECT label FROM labels WHERE profile_id=? AND entry_type=? AND entry_id=?",
        (profile_id, entry_type, entry_id)
    ).fetchone()
    conn.close()
    return row["label"] if row else None


def get_label_with_reasoning(entry_type: str, entry_id: int,
                              profile_id: int = 1) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        """SELECT label, reasoning, COALESCE(label_source, '') AS label_source
           FROM labels WHERE profile_id=? AND entry_type=? AND entry_id=?""",
        (profile_id, entry_type, entry_id)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "label": row["label"],
        "reasoning": row["reasoning"] or "",
        "label_source": row["label_source"] or "",
    }


def get_all_labels(profile_id: int = 1) -> List[dict]:
    """Get all labels with entry data for this profile."""
    conn = get_db()
    rows = conn.execute("""
        SELECT l.entry_type, l.entry_id, l.label, l.reasoning, l.created_at, l.updated_at,
               COALESCE(l.label_source, '') AS label_source,
               e.title, e.description, e.content, e.source_type, e.source_name, e.source_category
        FROM labels l
        JOIN entries e ON l.entry_type = e.entry_type AND l.entry_id = e.entry_id
        WHERE l.profile_id = ?
        ORDER BY l.updated_at DESC
    """, (profile_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_labels_raw(profile_id: int = 1) -> List[dict]:
    """Get all labels without joining entries (for syncing)."""
    conn = get_db()
    rows = conn.execute("""
        SELECT entry_type, entry_id, label, reasoning,
               COALESCE(label_source, '') AS label_source, created_at, updated_at
        FROM labels
        WHERE profile_id = ?
        ORDER BY updated_at DESC
    """, (profile_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_label_count(profile_id: int = 1) -> int:
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM labels WHERE profile_id = ?", (profile_id,)
    ).fetchone()[0]
    conn.close()
    return count


def get_label_distribution(profile_id: int = 1) -> dict:
    conn = get_db()
    rows = conn.execute("""
        SELECT label, COUNT(*) as count FROM labels
        WHERE profile_id = ?
        GROUP BY label ORDER BY count DESC
    """, (profile_id,)).fetchall()
    conn.close()
    return {r["label"]: r["count"] for r in rows}


def get_all_reasoning_texts(profile_id: int = 1) -> List[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT entry_type, entry_id, label, reasoning
        FROM labels
        WHERE profile_id = ? AND reasoning IS NOT NULL AND reasoning != ''
        ORDER BY updated_at DESC
    """, (profile_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Embedding operations ────────────────────────────────────────────────────

def store_embedding(entry_type: str, entry_id: int, embedding_bytes: bytes):
    conn = get_db()
    conn.execute(
        "UPDATE entries SET embedding = ? WHERE entry_type = ? AND entry_id = ?",
        (embedding_bytes, entry_type, entry_id)
    )
    conn.commit()
    conn.close()


def store_embeddings_batch(updates: List[tuple]):
    """Batch store embeddings. Each tuple: (embedding_bytes, entry_type, entry_id)."""
    conn = get_db()
    conn.executemany(
        "UPDATE entries SET embedding = ? WHERE entry_type = ? AND entry_id = ?",
        updates
    )
    conn.commit()
    conn.close()


def get_entries_without_embeddings(limit: int = 500) -> List[dict]:
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM entries WHERE embedding IS NULL
        ORDER BY published_date DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_embedding_count() -> int:
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return count


def invalidate_all_embeddings():
    conn = get_db()
    conn.execute("UPDATE entries SET embedding = NULL")
    conn.commit()
    conn.close()


# ─── Model operations ────────────────────────────────────────────────────────

def save_model_record(version: int, accuracy: float, f1: float, precision: float,
                      recall: float, label_count: int,
                      feature_count: int, model_path: str, recipe_path: str = "",
                      recipe_quality: float = 0.0,
                      architecture: str = "tfidf",
                      profile_id: int = 1,
                      label_distribution: Optional[dict] = None) -> int:
    """Save a model training record and set it as active for this profile."""
    dist_json = "{}"
    if label_distribution is not None:
        dist_json = json.dumps(label_distribution, ensure_ascii=False)
    conn = get_db()
    conn.execute("UPDATE models SET is_active = 0 WHERE profile_id = ?", (profile_id,))
    conn.execute("""
        INSERT INTO models (profile_id, version, accuracy, f1_score, precision_score,
                           recall_score, label_count, feature_count, model_path,
                           recipe_path, recipe_quality, is_active, architecture,
                           label_distribution)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
    """, (profile_id, version, accuracy, f1, precision, recall, label_count,
          feature_count, model_path, recipe_path, recipe_quality, architecture,
          dist_json))
    model_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return model_id


def get_active_model(profile_id: int = 1) -> Optional[dict]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM models WHERE is_active = 1 AND profile_id = ? ORDER BY id DESC LIMIT 1",
        (profile_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_models(profile_id: int = 1) -> List[dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM models WHERE profile_id = ? ORDER BY version DESC",
        (profile_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_next_model_version(profile_id: int = 1) -> int:
    conn = get_db()
    row = conn.execute(
        "SELECT MAX(version) FROM models WHERE profile_id = ?", (profile_id,)
    ).fetchone()
    conn.close()
    return (row[0] or 0) + 1


# ─── Sync log ────────────────────────────────────────────────────────────────

def log_sync(direction: str, items_count: int, details: str = "",
             profile_id: Optional[int] = None):
    conn = get_db()
    conn.execute(
        "INSERT INTO sync_log (profile_id, direction, items_count, details) VALUES (?, ?, ?, ?)",
        (profile_id, direction, items_count, details)
    )
    conn.commit()
    conn.close()


def get_recent_syncs(limit: int = 20,
                     profile_id: Optional[int] = None) -> List[dict]:
    conn = get_db()
    if profile_id is not None:
        rows = conn.execute("""
            SELECT * FROM sync_log
            WHERE profile_id = ? OR profile_id IS NULL
            ORDER BY synced_at DESC LIMIT ?
        """, (profile_id, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sync_log ORDER BY synced_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Export / Import labels ──────────────────────────────────────────────────

def export_labels(profile_id: int = 1) -> List[dict]:
    """Export all labels with entry text for a .magnitu package."""
    conn = get_db()
    rows = conn.execute("""
        SELECT l.entry_type, l.entry_id, l.label, l.reasoning, l.created_at, l.updated_at,
               COALESCE(l.label_source, '') AS label_source,
               e.title, e.description, e.content, e.link, e.author,
               e.published_date, e.source_name, e.source_category, e.source_type
        FROM labels l
        LEFT JOIN entries e ON l.entry_type = e.entry_type AND l.entry_id = e.entry_id
        WHERE l.profile_id = ?
        ORDER BY l.updated_at DESC
    """, (profile_id,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        src = (d.pop("label_source", "") or "").strip()
        if src:
            d["source"] = src
        out.append(d)
    return out


def import_labels(labels_list: List[dict], profile_id: int = 1) -> dict:
    """Import labels from a .magnitu package into a profile. Merges (newer wins).
    Also upserts the associated entry data so retraining works immediately."""
    conn = get_db()
    imported = 0
    skipped = 0
    updated = 0

    conn.execute("BEGIN IMMEDIATE")
    try:
        existing_map = {}
        for row in conn.execute(
            "SELECT entry_type, entry_id, label, updated_at FROM labels WHERE profile_id = ?",
            (profile_id,)
        ):
            existing_map[(row["entry_type"], row["entry_id"])] = {
                "label": row["label"], "updated_at": row["updated_at"] or ""
            }

        for lbl in labels_list:
            entry_type = lbl.get("entry_type", "")
            entry_id   = lbl.get("entry_id")
            label      = lbl.get("label", "")
            lbl_updated = lbl.get("updated_at", "")
            if not entry_type or not entry_id or not label:
                skipped += 1
                continue
            reasoning = lbl.get("reasoning", "")
            label_src = (lbl.get("source") or lbl.get("label_source") or "").strip()
            key = (entry_type, entry_id)
            existing = existing_map.get(key)
            if existing:
                if lbl_updated > existing["updated_at"]:
                    conn.execute("""
                        UPDATE labels SET label=?, reasoning=?, label_source=?, updated_at=?
                        WHERE profile_id=? AND entry_type=? AND entry_id=?
                    """, (label, reasoning, label_src, lbl_updated,
                          profile_id, entry_type, entry_id))
                    updated += 1
                else:
                    skipped += 1
            else:
                conn.execute("""
                    INSERT INTO labels
                        (profile_id, entry_type, entry_id, label, reasoning, label_source,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (profile_id, entry_type, entry_id, label, reasoning, label_src,
                      lbl.get("created_at", ""), lbl_updated))
                imported += 1

            if lbl.get("title") is not None:
                conn.execute("""
                    INSERT INTO entries
                        (entry_type, entry_id, title, description, content,
                         link, author, published_date, source_name,
                         source_category, source_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entry_type, entry_id) DO UPDATE SET
                        title=excluded.title, description=excluded.description,
                        content=excluded.content, link=excluded.link, author=excluded.author,
                        published_date=excluded.published_date,
                        source_name=excluded.source_name,
                        source_category=excluded.source_category,
                        source_type=excluded.source_type
                """, (entry_type, entry_id, lbl.get("title", ""),
                      lbl.get("description", ""), lbl.get("content", ""),
                      lbl.get("link", ""), lbl.get("author", ""),
                      lbl.get("published_date", ""), lbl.get("source_name", ""),
                      lbl.get("source_category", ""), lbl.get("source_type", "")))

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return {"imported": imported, "skipped": skipped, "updated": updated}


# ─── Legacy model_profile wrappers (used by model_manager.py) ───────────────
# These delegate to the profiles table so model_manager stays unchanged.

def get_model_profile(profile_id: int = 1) -> Optional[dict]:
    """Return profile in the old model_profile shape for backward compat."""
    p = get_profile_by_id(profile_id)
    if not p:
        return None
    return {
        "model_name": p["display_name"],
        "model_uuid": p["model_uuid"],
        "description": p["description"],
        "created_at": p["created_at"],
        # expose profile_id so callers that need it can find it
        "profile_id": p["id"],
        "slug": p["slug"],
    }


def set_model_profile(model_name: str, model_uuid: str, description: str = "",
                      created_at: str = None, profile_id: int = 1):
    """Create or update a profile (legacy API used by model_manager.import_model)."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    if existing:
        conn.execute("""
            UPDATE profiles SET display_name=?, model_uuid=?, description=?
            WHERE id=?
        """, (model_name, model_uuid, description or "", profile_id))
    else:
        slug_val = slugify(model_name)
        conn.execute("""
            INSERT INTO profiles (id, slug, display_name, description, model_uuid, is_default)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (profile_id, slug_val, model_name, description or "", model_uuid))
    conn.commit()
    conn.close()


def update_model_profile(description: Optional[str] = None, profile_id: int = 1):
    if description is not None:
        conn = get_db()
        conn.execute(
            "UPDATE profiles SET description=? WHERE id=?", (description, profile_id)
        )
        conn.commit()
        conn.close()


def has_model_profile(profile_id: int = 1) -> bool:
    p = get_profile_by_id(profile_id)
    return p is not None


# Initialize on import
init_db()
