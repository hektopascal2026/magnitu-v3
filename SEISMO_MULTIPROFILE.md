# Seismo 0.6 — Multi-Profile Integration Brief

**For Seismo operators and Magnitu admins.**  
Describes how **path satellites** in Seismo 0.6 work with Magnitu v3’s multi-profile layout.

---

## Background

Magnitu supports multiple **profiles** — each profile has its own labels, trained model, and a dedicated Seismo **push target**. A typical setup:

| Magnitu profile | What it learns | Pushes to |
|-----------------|----------------|-----------|
| Default | General relevance | Mothership Seismo (`/` or root `index.php`) |
| Security | Security-focused | Path desk `https://host/security/index.php` |
| Digital | Digital policy | Path desk `https://host/digital/index.php` |

**What did not change (Magnitu ↔ Seismo API):**

- Same JSON formats, field names, and endpoint signatures (`magnitu_entries`, `magnitu_scores`, `magnitu_recipe`, `magnitu_labels`, `magnitu_status`).
- Magnitu still pulls **all entries** from the **mothership** only (global Settings URL + key).
- Each desk **receives** scores, keyword recipes, and label sync from the Magnitu profile mapped to that desk’s URL + API key.

**What changed in Seismo 0.6 (ops only):**

- Satellites are **path desks** on one codebase (`/<slug>/`), not separate PHP trees or `seismo-generator` bundles.
- Provisioning: **Settings → Satellites** (registry) + `bin/seismo-satellite-provision.sh <slug>` on the VPS.
- Removed: `seismo-generator`, JSON deployment download, `SEISMO_MOTHERSHIP_DB` / `SEISMO_SATELLITE_MODE` config knobs.

---

## What is a path satellite?

A path satellite is a **desk** served from the same Seismo install as the mothership:

| Layer | Mothership | Satellite (e.g. `/security/`) |
|-------|------------|-------------------------------|
| Code | `/var/www/seismo` | Same tree |
| Entries | `seismo` database | Cross-DB read from `seismo` |
| Scores, labels, favourites, Magnitu config | `seismo` | `seismo_<slug>` |
| Public URL | `https://host/` | `https://host/security/` |
| Cron / ingest | `refresh_cron.php` | Refresh triggers mothership ingest (shared secret) |

Each desk:

1. **Reads entries** from the mothership entries database — no separate scraper.
2. **Stores its own** scores, recipes, training labels, and Magnitu `api_key` in `seismo_<slug>`.
3. **Exposes the same** `magnitu_*` API as the mothership (authenticated with that desk’s key).

The mothership remains the **system of record for raw entries**. Topic-specific labels and scores live on the desk Magnitu targets for that profile.

**Magnitu credential rule:** each profile stores `seismo_url` and `api_key` together. Magnitu rejects **incomplete** pairs (URL without key or key without URL).

---

## Seismo: provision a desk (operator)

1. **Mothership admin** — **Settings → Satellites** → add slug (e.g. `security`), display name, optional brand accent, optional **Magnitu profile** hint (slug for your Magnitu profile — convention only, not read over the API).
2. **VPS (SSH)** — from the app root:
   ```bash
   sudo bin/seismo-satellite-provision.sh security
   ```
   Creates `seismo_security`, runs scores migrations, seeds `api_key` from the registry into `seismo_security.system_config`, writes `/<slug>/index.php` stub and assets symlink, marks registry `status=active`.
3. **Verify desk** — open `https://your-host/security/` (timeline, label training, Settings → Magnitu).
4. **Copy credentials for Magnitu** — on that desk: **Settings → Magnitu** → copy **Seismo API URL** and **API key** (do not use the mothership key for a satellite profile).

**Key rotation:** mothership **Settings → Satellites → Rotate key** updates the registry; re-run provision logic or update `seismo_<slug>.system_config` and Magnitu’s profile key together.

---

## Magnitu: wire a profile (operator)

1. **Global Settings** — mothership `index.php` URL + API key (entry pull for all profiles).
2. **Profiles** — add profile (slug often matches Seismo desk slug, e.g. `security`).
3. **Profile Settings → Push target** — paste desk URL and API key from step 4 above, e.g. `https://your-host/security/index.php`.
4. **Test push target** — confirms `magnitu_status`; optional `accent_color` from Seismo registry accent tints the Magnitu tab.
5. **Sync → Label → Train → Push** — entries from mothership; labels pull/push and scores/recipe push go to the desk.

Leave push URL and key **blank** on a profile to use the mothership for labels and pushes (with a UI warning).

---

## Required API endpoints

No new endpoints. All desks implement the same contract:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `?action=magnitu_entries` | GET | Return entries (`type`, `since`, `limit`) |
| `?action=magnitu_scores` | POST | Batch scores |
| `?action=magnitu_recipe` | GET/POST | Recipe read/write |
| `?action=magnitu_labels` | GET/POST | Labels dump / upsert |
| `?action=magnitu_status` | GET | Health + stats |

Auth: `?api_key=...` query parameter (or `Authorization: Bearer` — Magnitu uses query params). Use the **desk** key for satellite targets, not the mothership key.

Optional on `magnitu_status`: **`accent_color`** (`#rrggbb` / `#rgb`) from Seismo `brand_accent` / desk branding.

`calendar_event` (Leg) is part of the same contract as feed / email / lex on Seismo 0.6+; Magnitu pulls and scores it when present.

---

## Score and recipe JSON formats

Unchanged. Desks must accept and store them like the mothership.

### Score object (`magnitu_scores`)

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

### Recipe (`magnitu_recipe`)

```json
{
  "version": 5,
  "classes": ["investigation_lead", "important", "background", "noise"],
  "class_weights": [1.0, 0.66, 0.33, 0.0],
  "keywords": { "leaked documents": {"investigation_lead": 0.82} },
  "source_weights": { "rss": {"important": 0.2} },
  "alert_threshold": 0.75
}
```

---

## Data flow

```
Mothership Seismo (https://host/)
│
│  ingest → entries in DB `seismo`
│
├─► Magnitu pull entries (global mothership URL + key; all profiles)
│         ↓
│   label + train per Magnitu profile
│         ↓
├─◄ Default profile → push scores / recipe / labels to mothership
│
Path desk (https://host/security/)
│
├─  reads entries from `seismo` (shared)
├─◄ Security profile → push scores / recipe / labels to desk URL + key
│
└─  readers use security-specific scores and labels in `seismo_security`
```

Entry IDs are **shared** across mothership and desks (same `entry_type` + `entry_id`). Magnitu never duplicates entry storage on Seismo; it caches entries locally once.

---

## Magnitu behaviour summary (`sync.py`)

| Operation | Target |
|-----------|--------|
| Pull entries | Global mothership only |
| Pull labels | Profile push target, or mothership if URL+key blank |
| Push scores, recipe, labels | Profile push target, or mothership if blank |
| Incomplete URL/key | `ValueError` — no mixing mothership key with desk URL |

---

## Legacy note (Seismo 0.5)

Older docs described **separate PHP installs**, `seismo-generator`, and config options `SEISMO_MOTHERSHIP_DB` / table-prefix sharing. Magnitu v3 did not need code changes for 0.6 path satellites — only URLs and keys differ. If you still run 0.5-style installs, use that instance’s full `index.php` URL and its Magnitu API key the same way; the HTTP contract is identical.

---

## Checklist

**Seismo**

- [ ] Satellite registered in **Settings → Satellites**
- [ ] `bin/seismo-satellite-provision.sh <slug>` completed
- [ ] Desk opens at `/<slug>/`
- [ ] Desk **Settings → Magnitu** shows URL + key

**Magnitu**

- [ ] Global mothership URL + key (entries)
- [ ] Profile push target URL + key (desk)
- [ ] Test push target OK
- [ ] Sync → Train → Push

**Isolation**

- [ ] Each topic profile that needs separate labels/scores has its **own** desk URL+key
- [ ] Mothership-only profiles have **both** push fields blank
