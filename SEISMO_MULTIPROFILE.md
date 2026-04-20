# Seismo — Multi-Profile Integration Brief

**For the Seismo developer.**  
This document describes the changes needed on the Seismo side to support Magnitu's new multi-profile architecture.

---

## Background: What Changed in Magnitu

Magnitu now supports multiple **profiles** — each profile has its own labels, trained model, and a dedicated Seismo push target. The typical setup will be:

| Profile | What it learns | Pushes to |
|---------|----------------|-----------|
| Default | General relevance (as before) | Mothership Seismo |
| Security | Security-focused | Lightweight Seismo "security" |
| Digitalthemen | Digital policy focus | Lightweight Seismo "digital" |

**What did not change:**

- The API contract between Magnitu and Seismo is **identical**. No changes to JSON formats, field names, or endpoint signatures.
- Magnitu still pulls all entries from the **mothership** Seismo (one pull source, as before).
- Each lightweight Seismo only **receives** scores and recipes — it does not need its own scraper.

---

## What is a "Lightweight Seismo"?

A lightweight Seismo is a **separate PHP installation** (running the same Seismo codebase) that:

1. **Reads entries from the mothership's database** — no independent scraping, no separate entry tables.  
2. **Maintains its own scoring and recipe tables** — scores and recipes from Magnitu's topic profile land here, independently of the mothership.
3. **Has its own API key** — so Magnitu can authenticate when pushing scores/recipes.
4. **Exposes the same API endpoints** as the mothership (`magnitu_entries`, `magnitu_scores`, `magnitu_recipe`, `magnitu_status`, `magnitu_labels`).

The mothership remains the system of record for entries, labels, and the global model. Lightweight instances are purely **consumers** of Magnitu scores and the entry pool.

---

## Implementation Options (for the Seismo developer to choose)

### Option A — Shared Database, Separate Table Prefix (recommended)

Same MySQL server, same host, but the lightweight instance uses a different table prefix (e.g. `seismo_security_` instead of `seismo_`).

- `seismo_security_entries` → **alias/view** pointing to the mothership's `seismo_entries` table (read-only)
- `seismo_security_scores`, `seismo_security_recipes`, `seismo_security_api_keys` → own tables

Seismo's `config.php` would accept a new directive:

```php
define('SEISMO_ENTRY_SOURCE', 'mothership');  // or 'local' (default)
define('SEISMO_MOTHERSHIP_PREFIX', 'seismo_'); // table prefix to read entries from
```

When `ENTRY_SOURCE = 'mothership'`, `magnitu_entries` reads from the mothership prefix, all other writes go to the local prefix.

### Option B — Separate MySQL Database, Cross-DB Queries

The lightweight instance has its own database (`seismo_security`) on the same MySQL server. Entries are read via a cross-database query:

```sql
SELECT * FROM mothership.seismo_entries WHERE ...
```

Configured in Seismo via:
```php
define('SEISMO_MOTHERSHIP_DB', 'seismo_main'); // mothership DB name, empty = use own DB
```

### Option C — Full Copy via Magnitu API (simplest, no DB sharing)

The lightweight Seismo runs fully independently — it receives entries via the `magnitu_entries` pull from the mothership (which Magnitu also calls during sync). No shared DB needed.

- All entry tables are local copies, refreshed on Magnitu's pull schedule.
- Simplest to deploy, least efficient (duplicate storage).

---

## Required API Endpoints

The lightweight Seismo instance needs to respond to the **same endpoints** as the mothership. No new endpoints are needed. For reference:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `?action=magnitu_entries` | GET | Return entries (params: `type`, `since`, `limit`) |
| `?action=magnitu_scores` | POST | Accept batch scores from Magnitu |
| `?action=magnitu_recipe` | GET/POST | GET current recipe; POST new recipe |
| `?action=magnitu_labels` | GET/POST | GET all labels; POST upsert labels |
| `?action=magnitu_status` | GET | Connectivity check + stats |

All requests include `?api_key=...` as a query parameter — the API key is the one configured for this lightweight instance.

---

## Score and Recipe JSON Formats

These are **unchanged from the existing contract**. The lightweight Seismo must accept and store them identically to the mothership.

### Score object pushed to `magnitu_scores`

```json
{
  "entry_type": "feed_item",
  "entry_id": 123,
  "relevance_score": 0.87,
  "predicted_label": "investigation_lead",
  "explanation": {
    "top_features": [{"feature": "...", "weight": 0.4, "direction": "positive"}],
    "confidence": 0.87,
    "prediction": "investigation_lead"
  }
}
```

Full push payload:
```json
{
  "scores": [...],
  "model_version": 5,
  "model_meta": {
    "model_name": "security-model",
    "model_uuid": "abc123",
    "model_version": 5,
    "accuracy": 0.78,
    "f1_score": 0.71,
    "label_count": 320,
    "architecture": "transformer"
  }
}
```

### Recipe pushed to `magnitu_recipe`

```json
{
  "version": 5,
  "classes": ["investigation_lead", "important", "background", "noise"],
  "class_weights": [1.0, 0.66, 0.33, 0.0],
  "keywords": {
    "leaked documents": {"investigation_lead": 0.82},
    "press release": {"noise": 0.45}
  },
  "source_weights": {
    "rss": {"important": 0.2}
  },
  "alert_threshold": 0.75
}
```

Seismo evaluates this in `scoreEntryWithRecipe()` in `config.php`. **No changes to that function are needed** — the format is identical.

---

## Setup Checklist for a Lightweight Seismo Instance

1. **Clone / copy the Seismo codebase** to a new path on the webspace (e.g. `/var/www/seismo_security/`).

2. **Create a new database or table prefix** for the instance's own tables:  
   - `scores`, `recipes`, `api_keys`, `labels` tables need their own home.  
   - Entries can be shared from the mothership (Options A/B) or replicated (Option C).

3. **Configure `config.php`** for this instance:  
   - Database connection (own DB or cross-DB to mothership)
   - `ENTRY_SOURCE` / `MOTHERSHIP_PREFIX` if implementing Option A/B
   - **Generate a fresh API key** for this instance — do not reuse the mothership key.

4. **Register the API key** in the Seismo admin panel (or directly in the DB) for this instance.

5. **Test connectivity** from Magnitu:  
   In Magnitu → Profile Settings (Settings page for the relevant profile), enter the lightweight Seismo URL and API key, then click **Test Connection**.

6. **First push**: Train the profile's model (or import one), then click **Push to Seismo** on the labeling page. Magnitu will push scores, labels, and recipe to this instance.

---

## Data Flow Summary

```
Mothership Seismo (netmind.ch)
│
│  scrapes/receives entries
│  stores all raw content
│
├─► Magnitu pulls entries (all profiles share this)
│         ↓
│   Magnitu labels + trains per profile
│         ↓
├─◄ Default profile pushes scores/recipe/labels
│
│
Lightweight Seismo — "Security" (webspace)
│
├─  reads entries FROM mothership DB (shared table or view)
├─◄ Security profile pushes scores/recipe
│
└─  its own consumers (readers, alerts) use security-specific scores
```

---

## What the Mothership Does NOT Need to Change

- No changes to existing endpoints or PHP logic.
- No changes to the recipe evaluation function `scoreEntryWithRecipe()`.
- No changes to the score/label JSON formats.
- The mothership continues receiving pushes from Magnitu's **default profile** exactly as before.

---

## Questions / Coordination Needed

1. **Which implementation option for entry sharing?** Recommend Option A (shared table prefix, same DB server) for simplicity and no data duplication.

2. **API key provisioning**: How are API keys generated in Seismo? The lightweight instance needs its own key registered before Magnitu can push to it.

3. **Label sync on lightweight instances**: Currently, `magnitu_labels` GET/POST is used to sync labels back. For lightweight instances, labels are profile-specific (e.g. security labels). Do the lightweight instances need to serve `magnitu_labels` at all, or should label sync go to the mothership only?  
   *Suggested answer*: Lightweight instances only need to accept `magnitu_scores` and `magnitu_recipe`. Label sync (`magnitu_labels` GET/POST) can be pointed at the mothership for all profiles — labels are global training data, not per-topic. Magnitu can be configured this way without any code change.

4. **Deployment path**: Should each lightweight instance have its own subdomain (e.g. `seismo-security.netmind.ch/index.php`) or run as subdirectories of the same domain?
