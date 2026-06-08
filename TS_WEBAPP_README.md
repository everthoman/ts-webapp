# Thompson Sampling + GNINA Docking Web App

A FastAPI web app that drives this repo's **Thompson Sampling** (TS) search using
**GNINA docking** as the scoring function. You pick predefined **reactions + reagent
sets**; multiple reactions form a **linear synthetic route** (step 1 builds an
intermediate from its reagent sets, each later step reacts that intermediate with one
more reagent set), and **only the final product is docked and scored**.

Every enumerated final product is **filtered for PAINS and REOS** (reusing the
`ligprepper` SMARTS sets) and, optionally, **MW / logP ranges** *before* docking — these
are **hard filters**: a failing molecule is rejected without docking and never counted.
Ligand preparation (OpenBabel protonation at a target pH + RDKit ETKDGv3 embed / MMFF94s
minimize) is the same pipeline used by the companion GNINA docking web app.

It lets you screen large, un-enumerated combinatorial libraries against a protein target
without enumerating or docking the whole space — TS samples the reagents that matter.

---

## Features

- **Predefined reactions + reagent sets** chosen from a JSON catalog (`reactions.json`).
- **Multi-step linear routes** — chain reactions; only the final product is scored.
- **Two search strategies** — standard **Thompson Sampling** (argmax of posterior draws) or
  **Roulette Wheel** sampling with thermal cycling (Zhao et al. 2025); both share the same
  absolute dock budget so a head-to-head comparison is apples-to-apples.
- **GNINA docking objective** — `minimizedAffinity` (Vina, minimize, default) or
  `CNNaffinity` / `CNN_VS` (maximize).
- **CPU-only by default** — `cnn_scoring=none` runs pure Vina docking on CPU (no GPU).
  Docks run **concurrently** (the CPU budget is split across parallel gnina processes);
  CNN modes use a GPU when selected.
- **Pre-dock filtering** — PAINS, REOS, MW range, logP range (all optional, all hard filters).
- **Pre-flight check** — sample products before committing, reporting filter pass-rate and
  MW/logP spread, and flagging a route that builds *no* products (reaction incompatibility)
  vs one rejected by filters.
- **Reagent-aware route building** — each extension reaction is annotated with a live
  *chain rate* (how often it actually reacts onto the previous step's products), so
  chemically incompatible chains are obvious up front.
- **SMILES score cache** — re-sampled products are never re-docked.
- **Live progress** over Server-Sent Events; a **top-scoring product gallery that updates
  during the run**; **cancellable** runs; **job history** with one-click rerun.
- **Downloadable results** — ranked unique products (`results.csv`) and best poses (`poses.sdf`).
- Usable **without the web app** — `GninaEvaluator` and `RouteSampler` are importable.

---

## Architecture

```
┌────────────────────────────┐
│ templates/index.html       │  route builder, filters, live SSE log
└──────────────┬─────────────┘
               │ POST /run  (receptor PDB + optional reference SDF + JSON config)
               ▼
┌────────────────────────────┐    RouteSampler (route_sampler.py)
│ ts_webapp.py  (FastAPI)    │──▶ warm-up + search (thompson_sampling.py)
│  background job + SSE log   │       │
└────────────────────────────┘       ▼  per final product
                              GninaEvaluator (gnina_evaluator.py)
                              PAINS/REOS/MW/logP filter ─▶ ligand prep ─▶ gnina dock
```

### Components

| File | Role |
|---|---|
| `ts_webapp.py` | FastAPI app + in-process job manager (one job at a time, SSE log stream). |
| `gnina_evaluator.py` | `GninaEvaluator` (TS scorer: docks with GNINA, caches by SMILES, keeps best poses), `MolFilters` (PAINS/REOS/MW/logP), ligand-prep helpers. |
| `route_sampler.py` | `RouteSampler` — multi-step linear route on top of `ThompsonSampler`. |
| `reactions.json` | Predefined reaction catalog + reagent-set registry. |
| `templates/index.html` | Single-page UI (no build step; CSS/JS inline). |
| `run_webapp.sh` | Launcher (`conda run -n ts_gnina uvicorn …`). |
| `ts-gnina-webapp.service` | systemd unit for persistent hosting. |

Builds on the existing TS engine: `thompson_sampling.py`, `reagent.py`,
`disallow_tracker.py`, `ts_utils.py`, `evaluators.py`.

### How it works

1. **Enumerate the route.** TS samples one reagent per reagent set. `RouteSampler` runs
   step 1's reaction on its reagents → intermediate; each later step reacts the
   intermediate with its new reagent → … → **final product**.
2. **Filter.** The final product is checked against MW/logP ranges, then PAINS, then REOS.
   Any failure → the product is rejected (`NaN` score), not docked, and skipped by TS.
3. **Prepare + dock.** Survivors are protonated (OpenBabel, target pH), embedded in 3D
   (RDKit ETKDGv3 + MMFF94s), and docked with GNINA in the binding box. The best pose's
   score (per the chosen field) is returned.
4. **Learn.** A warm-up (randomly sampling each reagent a few times to build priors) then a
   search loop that preferentially samples reagents that produced good scores. Two selection
   rules are available: **Thompson Sampling** picks the argmax of a posterior draw per
   reagent; **Roulette Wheel** (RWS) picks by Boltzmann-weighted roulette with a thermal-
   cycled temperature and a Boltzmann-weighted posterior mean — better diversity/coverage on
   large multi-component routes, while standard TS is the better pick for small libraries.

---

## Install

The repo uses **Python 3.10+** syntax, so use a 3.10/3.11 env (the pre-existing
`ts_vsflow` env is Python 3.9 and will **not** run this code):

```bash
conda create -n ts_gnina -c conda-forge python=3.11 rdkit numpy pandas tqdm
conda activate ts_gnina
pip install -r requirements-webapp.txt
```

External tools (must already be installed):

| Tool | Default location | Override |
|---|---|---|
| **gnina** binary ≥ 1.3 | `/opt/gnina/gnina.1.3.2` | `GNINA_PATH` |
| **OpenBabel** `obabel` ≥ 3.1 | on `PATH` | `OBABEL_PATH` |
| `PAINS.txt` / `REOS.txt` (from `ligprepper`) | `/opt/webapps/ligprepper` | `LIGPREPPER_DIR` |

---

## Run

### Development

```bash
./run_webapp.sh           # http://localhost:5014  (uses the ts_gnina env)
```

### As a systemd service (persistent, auto-restart, survives reboot)

A unit file `ts-gnina-webapp.service` is provided (runs on `0.0.0.0:5014`, reachable at
`http://130.237.250.75:5014`). Install it:

```bash
pkill -f "uvicorn ts_webapp:app"          # stop any temporary instance first
sudo cp /opt/webapps/TS/ts-gnina-webapp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ts-gnina-webapp
sudo systemctl status ts-gnina-webapp --no-pager
```

Logs: `journalctl -u ts-gnina-webapp -f`. Restart after code changes:
`sudo systemctl restart ts-gnina-webapp`.

### Firewall (KTH network only)

```bash
sudo ufw allow from 130.237.0.0/16 to any port 5014 proto tcp comment 'TS+GNINA webapp (KTH only)'
```

---

## Using the web UI

1. **Reaction route** — choose a starting reaction and a reagent set per component.
   Add extension steps to chain reactions (intermediate + one new reagent set); each
   extension reaction shows its live **chain rate** for the chosen upstream reagents.
2. **Receptor & binding site** — upload a prepared receptor PDB; define the box via a
   **reference ligand SDF** (autobox) or **XYZ coordinates + box edge**.
3. **GNINA** — score field (default `minimizedAffinity`), **CNN scoring** (default `none`
   = Vina/CPU; `rescore`/`all` use a GPU), exhaustiveness, poses, LigPrep pH.
4. **Filters** — toggle PAINS / REOS, set optional MW / logP ranges (blank = off).
   **Check filters** runs a pre-flight sample (pass-rate, MW/logP spread, route validity).
5. **Search** — pick the **method** (Thompson Sampling or Roulette Wheel), warm-up trials
   per reagent, the **search budget** (products to dock), parallel docks, seed, and — for
   RWS — min compounds/core and the resample-stop. GPU id is used only for CNN modes.
6. **Run** — watch the live log and the **top-product gallery filling in during the run**;
   **Cancel** to stop. When finished, download `results.csv` and `poses.sdf`. Past runs are
   listed in the job-history dropdown with one-click rerun.

---

## HTTP API

| Method & path | Purpose |
|---|---|
| `GET /` | The web UI (catalog injected). |
| `POST /run` | Multipart: `config` (JSON), `receptor` (PDB file), optional `reference` (SDF). Returns `{job_id}`. |
| `POST /preflight` | Form `config` (JSON). Samples products and returns filter pass-rate, MW/logP spread, and how many of the sampled combinations actually built a product. |
| `POST /extend-options` | Form `config` (JSON, upstream steps). Returns each extension reaction's live **chain rate** onto the upstream products. |
| `GET /jobs/{id}/status` | `{status, n_lines, n_results, error}`. |
| `GET /jobs/{id}/stream` | Server-Sent Events log stream; emits `event: end` with the final status. |
| `GET /jobs/{id}/top?n=` | Top-N ranked products (SVG + score + name); served **live from the running job** mid-run, else from `results.csv`. |
| `POST /jobs/{id}/cancel` | Request cancellation. |
| `POST /jobs/{id}/rerun` | Re-run a past job with its stored config. |
| `DELETE /jobs/{id}` | Delete a finished run's directory. |
| `GET /jobs/{id}/download/results.csv` | Ranked unique products. |
| `GET /jobs/{id}/download/poses.sdf` | Best docked pose per top product. |

`config` shape:

```json
{
  "steps": [{"reaction_id": "amide", "reagent_sets": ["primary_amines_100", "carboxylic_acids_100"]}],
  "binding_site": {"mode": "reference"},
  "gnina": {"score_field": "minimizedAffinity", "cnn_scoring": "none",
            "exhaustiveness": 8, "num_modes": 9, "autobox_add": 4.0, "ph": 7.4},
  "filters": {"pains": true, "reos": true, "mw": [200, 450], "logp": [null, 4]},
  "ts": {"method": "ts", "num_warmup_trials": 2, "num_ts_iterations": 200,
         "concurrency": null, "seed": null,
         "min_cpds_per_core": 50, "stop": 6000},
  "gpu_id": 0
}
```

- `ts.method` — `"ts"` (Thompson Sampling) or `"rws"` (Roulette Wheel). `num_ts_iterations`
  is the shared **search budget** (products to dock) for both methods.
- `ts.min_cpds_per_core` / `ts.stop` — RWS only (batch size; consecutive-resample stop).
- For `"binding_site"`, use `{"mode": "reference"}` (needs the `reference` SDF upload) or
  `{"mode": "coords", "center": [x, y, z], "box_size": 16.0}`.

---

## Outputs

**`results.csv`** — `score, SMILES, Name` (one row per unique final product, deduped and
ranked best-first per the chosen direction). `Name` is the underscore-joined reagent ids.

**`poses.sdf`** — the best docked pose per top product, written with `Chem.SDWriter`:

| Field | Meaning |
|---|---|
| `minimizedAffinity` | Vina binding affinity (kcal/mol, lower = better) |
| `CNNscore`, `CNNaffinity`, `CNN_VS` | GNINA CNN scores (present only with a CNN mode) |
| `SMILES` | canonical SMILES of the product (also the title) |
| `DockingRank` | 1 = best overall hit |

---

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `GNINA_PATH` | `/opt/gnina/gnina.1.3.2` | gnina binary |
| `OBABEL_PATH` | `obabel` | OpenBabel binary |
| `LIGPREPPER_DIR` | `/opt/webapps/ligprepper` | dir holding `PAINS.txt` / `REOS.txt` |
| `TS_DOCK_GPU` | `0` | default CUDA device (used only for CNN modes; overridable per run) |
| `TS_DOCK_CPU` | `N_CPU - 4` | `--cpu` threads per gnina dock (TS docks one mol at a time, so each gets most of the box) |
| `TS_RESERVED_CPU` | `4` | cores held back from each dock for the web server / TS loop |
| `TS_WEBAPP_PORT` | `5014` | server port |
| `TS_GNINA_ENV` | `ts_gnina` | conda env used by `run_webapp.sh` |

---

## Performance & resource use

- **Docking dominates runtime.** A CPU-only Vina dock is a few seconds; CNN modes add GPU
  time. The **SMILES cache** means re-sampled products are not re-docked.
- **Concurrent docking.** Warm-up and search dock in batches of `concurrency` molecules at
  once (the `TS_DOCK_CPU` budget is split across the parallel gnina processes); scores are
  recorded single-threaded afterwards so the sampling bookkeeping is identical to the
  sequential path. Set "Parallel docks" / `TS_CONCURRENCY`; blank = throughput-max default.
- **Cost estimate.** Warm-up performs roughly `Σ(reagents) × warmup_trials` docks; the
  search adds up to the **search budget** (`num_ts_iterations`) more, for both TS and RWS.
  The UI shows a live estimate. **Start small**: the 100-size reagent sets, 1–2 warm-up
  trials, tens-to-low-hundreds of budget — then scale up.

---

## Reaction catalog (`reactions.json`)

```json
{
  "reagent_sets": {
    "primary_amines_100": {"file": "data/primary_amines_100.smi", "count": 100, "label": "Primary amines (100)"}
  },
  "reactions": [
    {"id": "amide", "name": "Amide coupling (amine + acid)", "role": "start",
     "smarts": "[NH2:2][#6:1].[#6:4][C:3]([OH])=O>>[NH:2]([#6:1])[C:3]([#6:4])=O",
     "components": [
       {"label": "Primary amine",   "sets": ["primary_amines_100", "primary_amines_500"]},
       {"label": "Carboxylic acid", "sets": ["carboxylic_acids_100", "carboxylic_acids_500"]}
     ]}
  ]
}
```

- `reagent_sets`: `id → {file, count, label}`. Files are SMILES (`SMILES name` per line),
  paths relative to the repo root.
- `reactions`: each has `id`, `name`, `role`, `smarts`, and `components`.
  - `role: "start"` — used as step 1; consumes only reagents (no incoming intermediate).
  - `role: "extend"` — used as a later step; the running intermediate is automatically the
    **first** reactant of the SMARTS, so list only the additional new-reagent component(s).
  - `components` — one entry per **new** reagent the step consumes, each listing the allowed
    reagent-set ids (the order matches the reactant order in the SMARTS).

> Valid multi-step chains require the intermediate to expose the functional group the next
> reaction's SMARTS matches — and whether it does is **reagent-dependent** (the handle is
> only present if the chosen building blocks are difunctional). The UI's live **chain rate**
> per extension reaction (and the **Check filters** pre-flight) surface this empirically, so
> incompatible chains show a 0% rate rather than silently producing nothing.

---

## Programmatic use (no web app)

```python
from rdkit import Chem
from gnina_evaluator import GninaEvaluator, MolFilters
from route_sampler import RouteSampler

evaluator = GninaEvaluator({
    "receptor_path": "receptor.pdb",
    "reference_path": "reference_ligand.sdf",   # autobox; or pass center=(x,y,z)
    "score_field": "minimizedAffinity",          # minimize
    "cnn_scoring": "none",                        # CPU-only Vina
    "filters": MolFilters(use_pains=True, use_reos=True, mw_range=(200, 450)),
})

ts = RouteSampler(mode="minimize")               # minimize minimizedAffinity
ts.set_hide_progress(True)
ts.set_evaluator(evaluator)
ts.read_reagents(["data/primary_amines_100.smi", "data/carboxylic_acids_100.smi"])
ts.set_route([("[NH2:2][#6:1].[#6:4][C:3]([OH])=O>>[NH:2]([#6:1])[C:3]([#6:4])=O", 2)])

ts.warm_up(num_warmup_trials=2)
results = ts.search(num_cycles=200)              # [[score, smiles, name], ...]
evaluator.write_top_poses("poses.sdf", n=100)
```

`set_route` takes `[(reaction_smarts, num_new_reagents), …]`; the sum of `num_new_reagents`
must equal the number of reagent files. A single step reproduces the original
`ThompsonSampler` behaviour.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| *"Warm-up produced no scorable products…"* | Filters too tight — every product was rejected before docking. Loosen MW/logP or turn off PAINS/REOS. The message reports the filter/prep/dock breakdown. (The shipped amine/acid sets are very polar → low logP.) |
| All docks fail (`dock failures` high) | Check `GNINA_PATH`, the receptor PDB, and the binding box (reference SDF must overlap the pocket). |
| GPU seems unused | Expected — `cnn_scoring=none` runs CPU-only by design. Select `rescore`/`all` to use a GPU. |
| Port already in use | Another service holds 5014; set `TS_WEBAPP_PORT` to the next free port. |
| `useful_rdkit_utils` import warning | Harmless here — only `MLClassifierEvaluator` needs it; `MWEvaluator` falls back to RDKit. |

---

## Notes

- One TS job runs at a time (a process-wide lock) — docking is the bottleneck and the GPU /
  working directory are shared.
- Per-job working dirs and downloads live under `jobs/<id>/` (gitignored, not auto-pruned).
- The search loop tolerates library exhaustion on tiny libraries (reports partial results).

---

## Credits

Built on the **Thompson Sampling** implementation by Patrick Walters
([PatWalters/TS](https://github.com/PatWalters/TS), MIT) — see `README.md` and the paper
*"Thompson Sampling — An Efficient Method for Searching Ultralarge Synthesis-on-Demand
Databases"* (J. Chem. Inf. Model. 2024). The **Roulette Wheel** sampling option follows
Zhao et al., *"Enhanced Thompson sampling by roulette wheel selection for screening
ultralarge combinatorial libraries"* (J. Cheminform. 17:154, 2025; ref. implementation
[PatWalters/TS_2025](https://github.com/PatWalters/TS_2025)). The web app, GNINA evaluator,
multi-step route sampler, and pre-dock filtering were added on top. Licensed under MIT
(`LICENSE`).
