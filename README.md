# Magnitu

Magnitu is a machine-learning relevance engine for Seismo.  
It learns your labeling decisions (`investigation_lead`, `important`, `background`, `noise`) and pushes scores + a lightweight keyword recipe back to Seismo for live ranking.

Release **3.5** (see `VERSION` in `config.py`). This tree adds **Gemini** synthetic labeling, the **multilingual E5** embedding stack with a **LogReg** classifier head, and richer export manifests—on the same Seismo contract and multi-profile layout as before.

## Current stack

- **Local model**: transformer embeddings (`intfloat/multilingual-e5-base` by default) + LogReg classifier
- **Seismo runtime**: keyword recipe evaluated in PHP (distilled from local model via knowledge distillation)
- **Data sync**: pull **entries** from global mothership Settings; pull **labels** from each profile’s satellite (or mothership if satellite URL/key blank); push scores/recipe/labels per profile
- **Multi-profile**: multiple topic profiles (e.g. security, digital policy) each with their own labels, model, and push target — sharing one entry pool

## Key Features

- **Multi-profile support**
  - Each profile trains its own model on its own labeled entries
  - Each profile can push to a different Seismo desk (path satellite or mothership)
  - All profiles share the same entries and embeddings — sync once, score differently
  - Profile switcher in the top bar; **Settings → Profiles** to add and manage profiles (`/profiles` redirects there)
  - URL structure: `/p/{slug}/` for labeling, model (overview + exports), settings
- **Smart labeling queue**
  - `uncertain` — entries the current model is most confused about
  - `conflict` — model and recipe disagree
  - `diverse` — underrepresented source categories
  - `new` — most recent unlabeled entries
  - Source filter: All / Legislation / News
  - Full keyboard shortcuts
- **Sync modes**
  - **Sync**: quick incremental pull from mothership
  - **Full Sync**: source-by-source backfill + repeated embedding passes until coverage is complete
  - Live progress bar (phase label + percentage + ETA) for sync, **train**, and push
- **Scoring quality**
  - **Temperature calibration**: probabilities are fit via out-of-fold logits on the training fold so they are less overconfident before being pushed
  - **Enriched embeddings**: `source_type`, `source_name`, and `source_category` are prepended to each entry's text fingerprint; long bodies use **chunk pooling** (several E5 windows, length-weighted mean) so lex/Leg statutory text beyond the first ~512 tokens still influences the vector
  - **Lead discovery blend** (optional, 0–0.25): gently emphasises `investigation_lead` probability in the relevance score pushed to Seismo
  - **Rank-normalized push scores**: on Push, relevance scores sent to Seismo are replaced with their percentile rank within the batch (ordering preserved; spreads scores across the full range instead of clustering near ~0.5). Local Top/Mismatch views still use the raw model composite
  - **Synthetic label down-weight** (default `synthetic_label_weight: 0.5`): confirmed Gemini labels count half as much as human labels during training and recipe distillation; set `1.0` to disable
- **Advanced training knobs** (Settings → Advanced training, all opt-in)
  - **Label time-decay** and **reasoning-weight boost** — stored **per profile** (each workspace can use different values)
  - **Legal-signal patterns** — **global** (one list for all profiles; changing them invalidates shared embeddings)
  - Full reference with defaults, tradeoffs, and tuning guidance below: [Training Settings Reference](#training-settings-reference)
- **Recipe distillation**
  - Unigrams, bigrams, trigrams; both positive and negative signals per class
  - Legal-template priors and reasoning-phrase boosts (1.5×, deduped per phrase)
  - Knowledge distillation for transformer models (TF-IDF student learns from transformer soft labels; human labels weighted 3×)
  - Recipe export tuned against a **PHP-parity scorer** (once-per-keyword hit, synopsis-only lex/Leg text, accent-preserving tokens) so normalization and cap optimization match what Seismo actually runs
  - Per-profile model/recipe files (`model_p{id}_v{n}.joblib`, `recipe_p{id}_v{n}.json`) — multi-profile desks cannot overwrite each other's classifiers
- **Explainability**
  - Per-entry explanation showing top weighted features
  - Dashboard: learned legal-phrase patterns with impact scores
- **Model portability**
  - Export / import `.magnitu` packages (zip with model, labels, recipe, calibration)
  - Fork: export current model as a new identity for redistribution
- **Gemini synthetic labeling** (optional): background batch jobs from the Smart Queue; labels stored with reasoning for training (`GEMINI_API_KEY` in `.env`)

## Typical Workflow

1. **Sync** (or **Full Sync** on first run / after gaps)
2. **Label** entries on the labeling page
3. **Train**
4. **Push to Seismo**
5. Review **Top 30** and correct errors — these corrections are the highest-value training data

Repeat. Each cycle sharpens the model.

## Multi-Profile Workflow

```
Mothership Seismo → Magnitu pull (all profiles share entries)
                 ↓
         Profile: Default    Profile: Security    Profile: Digital
         labels + model       labels + model      labels + model
              ↓                    ↓                   ↓
         push to          push to path desk    push to path desk
         mothership        /security/          /digital/
```

To add a profile: **Settings** (for any profile) → **Profiles** → **Add profile** → name, optional **push target** URL + API key.

### Mothership → path satellite → Magnitu (canonical flow)

Seismo **0.6** uses **path satellites**: one codebase, desks at `https://host/<slug>/`, shared entries DB, per-desk scores DB `seismo_<slug>`. See `SEISMO_MULTIPROFILE.md` for the full brief.

1. **Seismo (mothership)** — **Settings → Satellites** → add slug (e.g. `security`), display name, optional accent. No JSON bundle or `seismo-generator`.
2. **VPS** — `sudo bin/seismo-satellite-provision.sh <slug>` (creates `seismo_<slug>`, desk stub, seeds Magnitu API key).
3. **Credentials** — Open `https://host/<slug>/` → **Settings → Magnitu** → copy **Seismo API URL** and **API key** for that desk.
4. **Magnitu** — Add a **profile** (slug often matches the desk). Paste URL + key under **Push target** (e.g. `https://host/security/index.php`). Training stays local (`.joblib`); Seismo receives **scores**, **recipe**, and **labels** only.
5. **Sync** — **Entries**: always from **global Mothership Connection**. **Labels**: from the desk when both push-target fields are set; otherwise mothership (UI warns).
6. **Push** — Scores, recipe, and labels go to that profile’s push target.

Optional **accent**: Seismo exposes `accent_color` on **`magnitu_status`** from the desk’s brand accent. Magnitu stores it after **Test push target** or **Push**. Entry IDs match the mothership (`entry_type` + `entry_id`) so labels and scores align.

### Seismo UI: what Magnitu sees (highlights vs training labels)

On a **satellite** (or mothership), the **Magnitu highlights** page lists entries whose score is Magnitu-sourced and above the alert threshold. Entry cards can show a **star** (favourite). Keep these two ideas separate:

| Action in Seismo | Stored in | Synced by Magnitu? |
|------------------|-----------|-------------------|
| **Star** on timeline / highlights | `entry_favourites` — **local to that Seismo instance** | **No.** Favourites never use the `magnitu_*` API. Starring on a satellite updates only that satellite’s database. |
| **Training labels** (`investigation_lead`, `important`, `background`, `noise`) on the **Label** tab | `magnitu_labels` — local DB table on that instance | **Yes**, via **`magnitu_labels`** GET (pull) and POST (push), scoped to the profile’s push target. |

So: work you want in Magnitu’s training loop (pull → train → push) belongs in the **Label** tab, not in stars on Highlights.

**HTTP routing in this repo (`sync.py`):**

- **Pull entries** — always the **global** mothership URL + API key from Settings (shared entry pool for every profile).
- **Pull labels**, **push scores**, **push recipe**, **push labels** — use the **profile’s** push target: when **both** `seismo_url` and `api_key` are set on the profile, every one of those calls goes to that pair (typically your **satellite**). When **both** are blank, Magnitu uses the global mothership for those operations. Setting **only one** of URL or key is rejected (`ValueError`) so a satellite URL is never mixed with a mothership key, or the reverse.

After provisioning, paste the desk **push URL** and **API key** (from that desk’s Seismo **Settings → Magnitu**) into **that** Magnitu profile’s push target. **Sync** pulls labels from the desk when configured; **Push** writes scores, recipe, and labels back to the same URL.

## Install from scratch (native)

**Requirements:** macOS or Linux, **Python 3.9+**, **git**, and a reachable **Seismo** instance with a **Magnitu API key** (Seismo → *Settings* → *Magnitu*). On macOS, **Xcode Command Line Tools** provide Python and git; `install/bootstrap.sh` can prompt you to install them if missing.

**1. Clone and run the installer**

```bash
git clone https://github.com/hektopascal2026/magnitu-v3.git
cd magnitu-v3
bash install/bootstrap.sh
```

You can use any directory name instead of `magnitu-v3`. If you already have the repo, run `bash install/bootstrap.sh` from the repository root; the script uses the current tree and does not clone again.

**2. What `bootstrap.sh` does**

- Creates a **`.venv`** in the repo and `pip install -r requirements.txt`
- Prompts for **mothership Seismo URL** (full `index.php` URL) and **API key**, then writes **`magnitu_config.json`**
- Optionally **resets** a leftover `magnitu.db` for a clean database
- **Tests** the Seismo connection
- Optionally walks through **model profile** creation, **importing a `.magnitu` file**, or **skip** (finish in the browser)

**3. Start the app**

```bash
./start.sh
```

Open **`http://127.0.0.1:8000`**. The first visit sends you to **`/setup`** to name your first profile if you have not already created one in the installer.

**Optional — desktop window:** same app in a **native window** (WebView) instead of using your default browser. After bootstrap, run `pip install -r requirements-desktop.txt` once, then **`./start_desktop.sh`** (or **`python desktop.py`** with the venv). If something is already listening on port 8000, only the window opens; otherwise a server is started and stopped when you close the window.

**macOS — `Magnitu.app`:** with the clone at **`~/Applications/magnitu3`**, **`install/bootstrap.sh`** also adds **Magnitu** to **`~/Applications`** and the **Desktop** (symlinks to `install/Magnitu.app` in the repo). Each launch runs a **git fetch / fast-forward on `main`** (same idea as `start.sh`), then the desktop window. If launch fails, you get an **alert** and a **log** (usually **`magnitu3/.magnitu_desktop_last.log`**) to send for debugging.

**Data on disk (default):** with no `MAGNITU_DATA_DIR` override, the database (`magnitu.db`), config (`magnitu_config.json`), and `models/` live in a **per-user data directory outside the git clone** — macOS: `~/Library/Application Support/Magnitu`, Linux: `~/.local/share/magnitu` (or `$XDG_DATA_HOME/magnitu`), Windows: `%LOCALAPPDATA%\\Magnitu`. This keeps secrets and local state out of the repository working tree. Override with env **`MAGNITU_DATA_DIR`**. On first start after an upgrade, if those files still sit next to `main.py`, Magnitu **moves** them into the new location automatically.

**Optional — Gemini synthetic labeling:** copy **`.env.example`** to **`.env`**, set **`GEMINI_API_KEY`**, and optionally **`GEMINI_MODEL`**. Do not commit `.env`.

**Alternative one-liner (no prior clone):** the script can clone into **`~/magnitu`** if you run it from a download/curl flow (see the header in `install/bootstrap.sh`). After that, use `~/magnitu/start.sh`. When you self-clone to e.g. `magnitu-v3`, use `./start.sh` in that folder instead.

**Update:** `cd` to your clone and `git pull` (or let `./start.sh` fast-forward on `main` if configured).

## Run with Docker

Magnitu ships with CPU and NVIDIA GPU container variants.

### CPU (works on Linux + Apple Silicon)

```bash
docker compose up --build
```

Open: `http://127.0.0.1:8000`

### NVIDIA GPU on Linux

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Docker data persistence

- App data is stored in the named volume `magnitu_data`:
  - config: `/app/data/magnitu_config.json` (global defaults; per-profile training overrides live in the DB)
  - database: `/app/data/magnitu.db`
  - models: `/app/data/models`
  - transformer cache: `/app/data/hf`
- On first start, Magnitu auto-creates `/app/data/magnitu_config.json` from `magnitu_config.example.json`.
- Set the mothership Seismo URL + API key in **Settings** after the first boot.

### Platform notes

- **NVIDIA Linux**: enable `GPU acceleration` in Settings to use CUDA.
- **Apple Silicon (M1/M2/M3)**: Docker Desktop Linux containers do not expose Metal/MPS, so Docker mode runs on CPU. Run Magnitu natively for GPU acceleration on macOS.

## Settings

Settings are split into **global** (apply to every profile) and **per-profile**. Global values live in `magnitu_config.json` in the data directory. Per-profile training overrides are stored in SQLite on each profile row (`training_settings` JSON) and merged over the global file when you train, score, or distill a recipe for that profile.

### Global (applies to all profiles)
- **Mothership Seismo URL + API key** — where entries are pulled from
- **Model architecture** — `transformer` or `tfidf`
- **Transformer model name** — HuggingFace model id (default `intfloat/multilingual-e5-base`)
- **GPU toggle** — CUDA/MPS acceleration when available
- **Legal-signal patterns** — shared across profiles (they are baked into the same cached embeddings for every entry)

### Per-profile
- **Push target URL + API key** — which Seismo instance receives this profile's scores and recipe. Leave blank to fall back to the mothership.
- **Training Settings** (minimum labels, recipe top keywords, alert threshold, lead discovery blend) — saved per profile when you click **Save** on that profile's Settings page. New profiles inherit global defaults until you change and save them there.
- **Advanced training** (label time-decay, decay floor, reasoning-weight boost, synthetic label weight) — also per profile. **Legal-signal patterns** remain global (see above).

## Training Settings Reference

Each knob below is tagged with its **default**, a sensible **range**, and guidance for when to raise or lower it. All changes apply to the *next* train; nothing already trained is re-weighted.

### Training Settings panel

**Minimum Labels to Train** — default `20`, range `5 – 500`  
Safety gate. Below 20 labels the transformer classifier can't generalise at all; around 50 classes start stabilising; 200+ is where minority classes (`investigation_lead`) get decent recall. Only lower this for debugging.

**Recipe Top Keywords** — default `200`, range `50 – 1000`  
How many features (words, bigrams, trigrams) the distilled recipe includes when pushed to Seismo. **Stored per profile** (each profile can use a different cap).
- **Tradeoff**: more keywords = broader coverage, but the recipe also grows (pushed to PHP on every update) and low-weight terms add noise to the Seismo live score. Fewer keywords = crisper signal, but niche terminology may fall below the cutoff.
- **Typical**: `150–300` for ~500 labels; `200–400` once you're past 1000 labels. Below 80 the recipe feels thin; above 600 you're mostly pushing noise.
- **Settings hint**: the Training panel shows a **heuristic** target (~200 scaling linearly toward **400** as label count approaches **2000**) when your current value is noticeably below that suggestion — optional guidance, not a requirement.
- **How to tune**: on the Dashboard, look at the **Learned Legal Patterns** panel. If the strongest positive signals look strong and the weakest look like filler, your cutoff is roughly right.

**Alert Threshold** — default `0.75`, range `0.0 – 1.0`  
Score above which Seismo highlights an entry as an alert.
- **Tradeoff**: too high → you miss borderline leads; too low → alerts become noise.
- **How to tune**: open the Top-30 page. Find the lowest score that still looks like something you'd want an alert on; set the threshold slightly below it.

**Lead discovery blend** — default `0`, range `0.0 – 0.25`  
Mixes extra `investigation_lead` probability into the pushed relevance score.
- `0` = pure composite (safest, most interpretable).
- `0.1` = gentle nudge — useful if the composite scoring feels conservative.
- `0.25` = aggressive — use only if you've verified on the Top-30 that genuine leads are getting scored mid-range.

### Advanced training panel

**Label time-decay half-life (days)** — default `0` (off), range `0 – 3650`  
Half-life for a label's training weight. `0` disables decay; `120` means a 4-month-old label counts half as much as a fresh one.
- **When to enable**: your editorial focus has shifted and old labels are dragging the model toward now-irrelevant topics.
- **Typical values**: `90–180` days for "editorial focus follows the news cycle"; `365+` for "gentle recency preference". Values under 30 days are rarely useful unless you're doing rapid topic pivots.
- **Caveat**: only meaningful once your labels span a decent time range (months, not weeks).

**Decay floor** — default `0.2`, range `0.0 – 1.0`  
Minimum weight after unlimited decay.
- `0.2` = old labels still carry a fifth as much as fresh ones (default, recommended).
- `0.0` = old labels vanish entirely — not recommended; you lose training data.
- `0.5` = "soft recency" — decay just nudges priorities, everything still counts substantially.

**Reasoning-weight boost** — default `1.0` (off), range `0.0 – 5.0`  
Multiplier for labels that carry a written reasoning note. Labels receive `sample_weight` on the transformer and TF-IDF paths by this factor.
- `1.0` = off; `1.3 – 1.5` = gentle boost (recommended starting point); `2.0` = strong; above `3.0` = experimental.
- **Heuristic**: if a minority of your labels have reasoning (say 10–30%), you can afford a stronger boost. If most labels have reasoning, stay near `1.0` — the boost stops being informative.
- The model never reads the reasoning text; only the *presence* of reasoning changes the label's weight. Reasoning phrases are also boosted in the distilled recipe (see About page).

**Synthetic label weight** — default `0.5`, range `0.0 – 1.0`  
Training multiplier for labels with `label_source = Gemini` (confirmed synthetic batch rows only; pending Accept rows are excluded from recipe reasoning boost).
- `0.5` = Gemini labels count half as much as human labels in the classifier and in distillation hard-label overrides.
- `1.0` = legacy behaviour (Gemini and human labels weigh equally).
- Use after large synthetic batches so human corrections stay authoritative.

**Legal-signal patterns** — default empty (list of strings), **global (all profiles)**  
Phrases (literal or regex) that, when matched in an entry's text, are:
1. prepended to the embedded input as `signals=<matched phrases>` so the transformer sees them as emphasized context, and
2. boosted in the distilled recipe pushed to Seismo (positive for `investigation_lead` and `important`).
- **How many**: **5 – 20** is the sweet spot. Fewer than 3 rarely moves the needle; more than 30 dilutes each pattern's contribution.
- **How to find good ones**:
  1. Open the Top-30 page filtered to Legislation and scan the highest-scoring lex items.
  2. Look for recurring phrases that distinguish investigation-worthy legal news from procedural boilerplate — e.g. exclusion language, third-country provisions, sector-specific triggers.
  3. Check the Dashboard's *Learned Legal Patterns* for phrases the model already considers strong positive signals — good candidates to lock in explicitly.
  4. Start with 5 obvious domain-specific phrases, train, push, observe; add more cautiously.
- **Cost**: changing this list invalidates *all* cached embeddings. Next sync does a full re-embed — plan for a few minutes of CPU/GPU time.
- **Example set for Swiss KMU/Export**: `Drittland`, `Binnenmarkt`, `EWR`, `CE-Kennzeichnung`, `Ursprungserzeugnis`, `Konformitätsbewertung`, `Marktüberwachung`, `Gleichwertigkeit`, `Zollkodex`, `Ursprungsregel`.

### When to retrain
Training is manual — click **Train** on the Label page (background job with progress bar). As a rule of thumb: retrain after every **10–20 new labels** during active labeling, or after any large labeling session. Training is cheap with cached embeddings (seconds to a few minutes; first E5 load or many missing embeddings takes longer).

## Scoring pipeline reference

Design notes and implementation plans live in `docs/scoring-fix-plan.md` and `docs/sharpness-fix-plan.md`. Summary of what ships in this tree:

| Area | Behaviour |
|------|-----------|
| **Local model** | E5 embeddings + LogReg head; temperature calibration on OOF logits |
| **Push to Seismo** | Percentile rank within the pushed batch (monotone; safe for sorting/thresholds) |
| **Recipe** | Distilled TF-IDF student → keyword JSON; PHP scorer parity for tuning; legal-template floors |
| **Multi-profile** | Separate `model_p*_*` / `recipe_p*_*` files per profile |
| **Long documents** | Chunk pooling at embed time (4 chunks news, 6 lex/Leg); re-embed after upgrade for full benefit |
| **Gemini labels** | `synthetic_label_weight` (default 0.5) in per-profile training settings |
| **Label sync** | Normalized timestamp compare; `label_source` preserved on remote merge |

Seismo evaluates the **recipe** on new/unscored rows; Magnitu-pushed **scores** take precedence when present. Recipe and push scores are related but not identical — rank normalization applies only to the pushed score batch.

## Database Migration

Magnitu migrates existing installations automatically on startup:
- Existing labels are assigned to the default profile (profile_id = 1)
- Existing model records are assigned to the default profile
- The old `model_profile` table is ported to the new `profiles` table
- The `profiles` table gains a `training_settings` JSON column for per-profile training overrides (empty until you save Training / Advanced knobs for a profile)
- No data is lost

## Notes

- Python 3.9 compatibility is preserved throughout.
- Each trained classifier is saved with a **calibration JSON** sidecar (`.calibration.json`). `.magnitu` exports include it as `calibration.json` so imports keep the same scoring behaviour.
- Seismo uses one active recipe per instance at a time. Last push wins.
- If you see recipe/model mismatch, run **Full Sync → Train → Push**.
- See `SEISMO_MULTIPROFILE.md` for Seismo 0.6 path-satellite setup and Magnitu wiring.
- See `docs/magnitu-entry-pills.md` for Seismo timeline source-pill styling (entry cards in Magnitu UI).
