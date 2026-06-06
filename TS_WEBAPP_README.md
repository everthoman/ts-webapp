# Thompson Sampling + GNINA Docking Web App

A FastAPI web app that drives this repo's Thompson Sampling (TS) search using
**GNINA docking** as the scoring function. You pick one or more **predefined
reactions + reagent sets**; multiple reactions form a **linear synthetic route**
(step 1 builds an intermediate from its reagent sets, each later step reacts that
intermediate with one more reagent set), and **only the final product is docked**.

Every enumerated final product is **filtered for PAINS and REOS** (reusing the
`ligprepper` SMARTS sets) and, optionally, **MW / logP ranges** *before* docking —
these are hard filters: a failing molecule is rejected without docking. Ligand
preparation (OpenBabel protonation at a target pH + RDKit ETKDGv3 embed / MMFF94s
minimize) is the same pipeline used by the GNINA web app.

```
┌────────────────────────────┐
│ templates/index.html       │  route builder, filters, live SSE log
└──────────────┬─────────────┘
               │ POST /run (receptor PDB + optional reference SDF + JSON config)
               ▼
┌────────────────────────────┐    RouteSampler (route_sampler.py)
│ ts_webapp.py  (FastAPI)    │──▶ warm-up + search (thompson_sampling.py)
│  background job + SSE log   │       │
└────────────────────────────┘       ▼  per final product
                              GninaEvaluator (gnina_evaluator.py)
                              PAINS/REOS/MW/logP filter ─▶ ligand prep ─▶ gnina dock
```

## Components

| File | Role |
|---|---|
| `ts_webapp.py` | FastAPI app + in-process job manager (one job at a time). |
| `gnina_evaluator.py` | `GninaEvaluator` (TS scorer that docks with GNINA, caches by SMILES, keeps best poses), `MolFilters` (PAINS/REOS/MW/logP), ligand-prep helpers. |
| `route_sampler.py` | `RouteSampler` — multi-step linear route on top of `ThompsonSampler`. |
| `reactions.json` | Predefined reaction catalog + reagent-set registry. |
| `templates/index.html` | Single-page UI. |
| `run_webapp.sh` | Launcher (`conda run -n ts_gnina uvicorn …`). |

## Install

The repo uses Python 3.10+ syntax, so use a 3.10/3.11 env:

```bash
conda create -n ts_gnina -c conda-forge python=3.11 rdkit numpy pandas tqdm
conda activate ts_gnina
pip install -r requirements-webapp.txt
```

External tools (must already be installed):
- **gnina** binary — default `/opt/gnina/gnina.1.3.2`, override with `GNINA_PATH`.
- **OpenBabel** `obabel` ≥ 3.1 on `PATH` (ligand protonation).
- `ligprepper`'s `PAINS.txt` / `REOS.txt` — default `/opt/webapps/ligprepper`, override with `LIGPREPPER_DIR`.

## Run

```bash
./run_webapp.sh           # http://localhost:5014
```

Then in the browser:
1. **Reaction route** — choose a starting reaction and a reagent set per component.
   Add extension steps to chain reactions (the intermediate + one new reagent set).
2. **Receptor & binding site** — upload a prepared receptor PDB; define the box via a
   **reference ligand SDF** (autobox) or **XYZ + box edge**.
3. **GNINA** — score field (default `minimizedAffinity`, minimize; `CNNaffinity` /
   `CNN_VS` maximize), CNN scoring mode, exhaustiveness, poses, LigPrep pH.
4. **Filters** — toggle PAINS / REOS, set optional MW / logP ranges (blank = off).
5. **Thompson Sampling** — warm-up trials per reagent, search iterations, GPU id.
6. **Run** — watch the live log; download `results.csv` (ranked unique products) and
   `poses.sdf` (best docked pose per top product) when finished. **Cancel** stops the run.

## Run as a systemd service

A unit file `ts-gnina-webapp.service` is provided (runs on `0.0.0.0:5014`,
reachable at http://130.237.250.75:5014). Install it:

```bash
pkill -f "uvicorn ts_webapp:app"          # stop any temporary instance first
sudo cp /opt/webapps/TS/ts-gnina-webapp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ts-gnina-webapp
sudo systemctl status ts-gnina-webapp --no-pager
```

Logs: `journalctl -u ts-gnina-webapp -f`. Restart after code changes:
`sudo systemctl restart ts-gnina-webapp`.

Firewall (KTH network only):
```bash
sudo ufw allow from 130.237.0.0/16 to any port 5014 proto tcp comment 'TS+GNINA webapp (KTH only)'
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `GNINA_PATH` | `/opt/gnina/gnina.1.3.2` | gnina binary |
| `OBABEL_PATH` | `obabel` | OpenBabel binary |
| `LIGPREPPER_DIR` | `/opt/webapps/ligprepper` | dir holding `PAINS.txt` / `REOS.txt` |
| `TS_DOCK_GPU` | `0` | default CUDA device (overridable per run in the UI) |
| `TS_DOCK_CPU` | `N_CPU - 4` | `--cpu` threads per gnina dock. TS docks one molecule at a time, so each dock gets most of the box. |
| `TS_RESERVED_CPU` | `4` | cores held back from each dock for the web server / TS loop |
| `TS_WEBAPP_PORT` | `5014` | server port |
| `TS_GNINA_ENV` | `ts_gnina` | conda env used by `run_webapp.sh` |

## Cost note

Docking dominates runtime (seconds per molecule on GPU). Warm-up performs roughly
`Σ(reagents) × warmup_trials` docks; each search iteration adds at most one more.
Scores are cached by canonical SMILES, so re-sampled products are not re-docked.
Start with the small reagent sets (100s), 1–2 warm-up trials, and tens to low-hundreds
of iterations, then scale up.

## Extending the catalog

Edit `reactions.json`:
- `reagent_sets`: `id → {file, count, label}` (SMILES files: `SMILES name` per line).
- `reactions`: each has `id`, `name`, `role` (`start` for step 1, `extend` for later
  steps), `smarts`, and `components` (one entry per **new** reagent the step consumes,
  each listing allowed reagent-set ids). For `extend` reactions the running intermediate
  is automatically the first reactant of the SMARTS; list only the additional reagent
  component(s). Valid chains require the intermediate to expose the functional group the
  next reaction matches.

## CLI use

`GninaEvaluator` and `RouteSampler` are usable without the web app, e.g. from a script
or notebook — build the `input_dict` for `GninaEvaluator`, call
`RouteSampler.set_route([(smarts, n_new_reagents), …])`, then `read_reagents`,
`warm_up`, and `search` as in `ts_main.run_ts`.
