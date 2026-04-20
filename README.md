# Magnitu

Magnitu is a machine-learning relevance engine for Seismo.  
It learns your labeling decisions (`investigation_lead`, `important`, `background`, `noise`) and pushes scores + a lightweight keyword recipe back to Seismo for live ranking.

Release **3.x** (see `VERSION` in `config.py`). This tree adds **Gemini** synthetic labeling, an on-disk **`.magnitu` library**, and richer export manifests—on the same Seismo contract and multi-profile layout as before.

## Current stack

- **Local model**: transformer embeddings (`xlm-roberta-base` by default) + MLP classifier
- **Seismo runtime**: keyword recipe evaluated in PHP (distilled from local model via knowledge distillation)
- **Data sync**: pull entries/labels from a mothership Seismo; push scores/recipe/labels per profile
- **Multi-profile**: multiple topic profiles (e.g. security, digital policy) each with their own labels, model, and push target — sharing one entry pool

## Key Features

- **Multi-profile support**
  - Each profile trains its own model on its own labeled entries
  - Each profile can push to a different (lightweight) Seismo instance
  - All profiles share the same entries and embeddings — sync once, score differently
  - Profile switcher in the top bar; **Settings → Profiles** to add and manage profiles (`/profiles` redirects there)
  - URL structure: `/p/{slug}/` for labeling, dashboard, model, settings
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
  - Live progress bar (phase label + percentage) for sync and push
- **Scoring quality**
  - **Temperature calibration**: probabilities are fit on a held-out validation slice so they are less overconfident before being pushed
  - **Enriched embeddings**: `source_type`, `source_name`, and `source_category` are prepended to each entry's text fingerprint; the embedding is invalidated when those fields change on sync
  - **Lead discovery blend** (optional, 0–0.25): gently emphasises `investigation_lead` probability in the relevance score pushed to Seismo
- **Advanced training knobs** (Settings → Advanced training, all opt-in)
  - **Label time-decay** — older labels count less at training time (configurable half-life, with a floor so ancient labels never fully vanish)
  - **Reasoning-weight boost** — labels that come with a written reasoning note count more during training
  - **Legal-signal patterns** — user phrases that get prepended to the embedded text as `signals=…` and boosted in the distilled recipe
  - Full reference with defaults, tradeoffs, and tuning guidance below: [Training Settings Reference](#training-settings-reference)
- **Recipe distillation**
  - Unigrams, bigrams, trigrams; both positive and negative signals per class
  - Legal-template priors and reasoning-phrase boosts
  - Knowledge distillation for transformer models (TF-IDF student learns from transformer predictions)
- **Explainability**
  - Per-entry explanation showing top weighted features
  - Dashboard: learned legal-phrase patterns with impact scores
- **Model portability**
  - Export / import `.magnitu` packages (zip with model, labels, recipe, calibration)
  - Fork: export current model as a new identity for redistribution
- **Gemini synthetic labeling** (optional): background batch jobs from the Smart Queue; labels stored with reasoning for training (`GEMINI_API_KEY` in `.env`)
- **Library**: list `*.magnitu` under the models directory and activate one per profile without leaving the UI

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
         push to          push to lightweight  push to lightweight
         mothership        Seismo "security"   Seismo "digital"
```

To add a profile: **Profiles** page → **Add Profile** → give it a name and (optionally) a dedicated push-target URL.

## Run locally

```bash
git clone https://github.com/hektopascal2026/magnitu-v3.git
cd magnitu-v3
bash install/bootstrap.sh
./start.sh
```

Use any directory name you like; `install/bootstrap.sh` detects an existing checkout when run from inside the repo.

Open: `http://127.0.0.1:8000`  
First run redirects to `/setup` to name your first profile.

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
  - config: `/app/data/magnitu_config.json`
  - database: `/app/data/magnitu.db`
  - models: `/app/data/models`
  - transformer cache: `/app/data/hf`
- On first start, Magnitu auto-creates `/app/data/magnitu_config.json` from `magnitu_config.example.json`.
- Set the mothership Seismo URL + API key in **Settings** after the first boot.

### Platform notes

- **NVIDIA Linux**: enable `GPU acceleration` in Settings to use CUDA.
- **Apple Silicon (M1/M2/M3)**: Docker Desktop Linux containers do not expose Metal/MPS, so Docker mode runs on CPU. Run Magnitu natively for GPU acceleration on macOS.

## Settings

Settings are split into **global** (apply to every profile) and **per-profile** (only affect the active profile). All global settings live in `magnitu_config.json` in the data directory; per-profile settings live in the database.

### Global (applies to all profiles)
- **Mothership Seismo URL + API key** — where entries are pulled from
- **Model architecture** — `transformer` or `tfidf`
- **Transformer model name** — HuggingFace model id (default `xlm-roberta-base`)
- **GPU toggle** — CUDA/MPS acceleration when available

### Per-profile
- **Push target URL + API key** — which Seismo instance receives this profile's scores and recipe. Leave blank to fall back to the mothership.
- **Advanced training knobs** below all sit at the global config level today (they affect every profile's next train); this is a deliberate simplification — revisit if a profile ever needs different weighting.

## Training Settings Reference

Each knob below is tagged with its **default**, a sensible **range**, and guidance for when to raise or lower it. All changes apply to the *next* train; nothing already trained is re-weighted.

### Training Settings panel

**Minimum Labels to Train** — default `20`, range `5 – 500`  
Safety gate. Below 20 labels the transformer classifier can't generalise at all; around 50 classes start stabilising; 200+ is where minority classes (`investigation_lead`) get decent recall. Only lower this for debugging.

**Recipe Top Keywords** — default `200`, range `50 – 1000`  
How many features (words, bigrams, trigrams) the distilled recipe includes when pushed to Seismo.
- **Tradeoff**: more keywords = broader coverage, but the recipe also grows (pushed to PHP on every update) and low-weight terms add noise to the Seismo live score. Fewer keywords = crisper signal, but niche terminology may fall below the cutoff.
- **Typical**: `150–300` for ~500 labels; `200–400` once you're past 1000 labels. Below 80 the recipe feels thin; above 600 you're mostly pushing noise.
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
Multiplier for labels that carry a written reasoning note. Labels are replicated (for MLP) or `sample_weight`-ed (for TF-IDF) by this factor.
- `1.0` = off; `1.3 – 1.5` = gentle boost (recommended starting point); `2.0` = strong; above `3.0` = experimental.
- **Heuristic**: if a minority of your labels have reasoning (say 10–30%), you can afford a stronger boost. If most labels have reasoning, stay near `1.0` — the boost stops being informative.
- The model never reads the reasoning text; only the *presence* of reasoning changes the label's weight.

**Legal-signal patterns** — default empty (list of strings)  
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
Training is manual — click **Train** on the Model page. As a rule of thumb: retrain after every **10–20 new labels** during active labeling, or after any large labeling session. Training is cheap with cached embeddings (seconds to a minute for most setups), so retraining often is the right move.

## Database Migration

Magnitu migrates existing installations automatically on startup:
- Existing labels are assigned to the default profile (profile_id = 1)
- Existing model records are assigned to the default profile
- The old `model_profile` table is ported to the new `profiles` table
- No data is lost

## Notes

- Python 3.9 compatibility is preserved throughout.
- Each trained classifier is saved with a **calibration JSON** sidecar (`.calibration.json`). `.magnitu` exports include it as `calibration.json` so imports keep the same scoring behaviour.
- Seismo uses one active recipe per instance at a time. Last push wins.
- If you see recipe/model mismatch, run **Full Sync → Train → Push**.
- See `SEISMO_MULTIPROFILE.md` for the brief to give the Seismo developer when setting up lightweight Seismo instances.
