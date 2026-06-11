# Thompson Sampling + GNINA Docking

A web app (and library) that screens large, **un-enumerated** combinatorial libraries
with **Thompson Sampling**, using **GNINA molecular docking** as the scoring function.

You pick predefined **reactions + reagent sets**; Thompson Sampling decides which reagents
to combine, each enumerated product is filtered (PAINS / REOS / MW / logP) and prepared in
3D, then docked with GNINA — and the search learns which reagents give the best binders
without ever enumerating or docking the whole library.

> Built on the Thompson Sampling implementation by Patrick Walters
> ([PatWalters/TS](https://github.com/PatWalters/TS)) and the paper
> [*"Thompson Sampling — An Efficient Method for Searching Ultralarge Synthesis-on-Demand
> Databases"*](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790) (J. Chem. Inf. Model. 2024).
> This repo adds a GNINA docking evaluator, structural/property pre-filtering, multi-step
> reaction routes, and a FastAPI web app on top of that engine.

---

## What this adds to vanilla Thompson Sampling

| Capability | Where |
|---|---|
| **GNINA docking as the scoring function** (Vina `minimizedAffinity` on CPU by default, or CNN scores on GPU) | `gnina_evaluator.py` → `GninaEvaluator` |
| **Pre-dock hard filters** — PAINS, REOS (reusing `ligprepper` SMARTS), MW range, logP range | `gnina_evaluator.py` → `MolFilters` |
| **Ligand prep** — OpenBabel protonation at pH + RDKit ETKDGv3 / MMFF94s 3D embed | `gnina_evaluator.py` |
| **Automatic deprotection before docking** — Fmoc/Boc/Cbz amines, tBu/Bn esters, Bpin boronates stripped so products are scored as their free forms | `gnina_evaluator.py` → `deprotect_smiles` |
| **Multi-step linear reaction routes** — chain reactions; only the final product is docked | `route_sampler.py` → `RouteSampler` |
| **Web app** — pick reactions/reagents, upload receptor, set filters, watch live progress, download results + poses | `ts_webapp.py`, `templates/index.html` |

---

## Quick start (web app)

```bash
# Python 3.10+ env (3.9 will not run this code)
conda create -n ts_gnina -c conda-forge python=3.11 rdkit numpy pandas tqdm
conda activate ts_gnina
pip install -r requirements-webapp.txt

./run_webapp.sh        # → http://localhost:5014
```

Needs the `gnina` binary (`GNINA_PATH`) and `obabel` on `PATH`. In the browser you choose a
reaction route + reagent sets, upload a receptor PDB and a reference-ligand SDF (autobox) or
XYZ box, toggle PAINS/REOS and optional MW/logP ranges, set the Thompson Sampling
warm-up/iterations, and run. You get a ranked `results.csv` and a best-pose `poses.sdf`.

**Full web-app documentation — install, deployment (systemd + ufw), HTTP API, output
fields, performance, reaction-catalog format, troubleshooting — is in
[`TS_WEBAPP_README.md`](TS_WEBAPP_README.md).**

---

## How Thompson Sampling works

Thompson Sampling is an active-learning strategy that balances exploitation and exploration.
It models the score distribution of each reagent as a normal distribution whose mean we are
estimating.

1. **Warm-up:** randomly sample (make + score a product with) each reagent _n_ times; build
   each reagent's prior from the mean/std of those scores.
2. **Search,** repeated for the requested number of iterations:
   - take a random pull from each reagent's distribution,
   - pick, for each reaction component, the reagent with the best pull,
   - build the product, score it (here: **dock it with GNINA**),
   - Bayesian-update the chosen reagents' distributions with the observed score.

Scores and SMILES for every product made are recorded; here, filtered or failed products
score `NaN` and are simply skipped (no prior update, not counted).

### Multi-step routes

When more than one reaction is selected they form a **linear synthetic route**: step 1 builds
an intermediate from its reagent sets, each later step reacts that intermediate with one more
reagent set, and **only the final product is docked/scored**. `RouteSampler.set_route([...])`
takes `[(reaction_smarts, num_new_reagents), …]`; a single step reproduces the original
single-reaction `ThompsonSampler`.

---

## Repository layout

**Docking / web app (added here)**
- `ts_webapp.py` — FastAPI app + job manager
- `gnina_evaluator.py` — `GninaEvaluator`, `MolFilters`, ligand prep, `deprotect_smiles`
- `route_sampler.py` — `RouteSampler` (multi-step routes)
- `reactions.json` — predefined reaction + reagent-set catalog
- `functional_groups.json` — FG vocabulary for the synthon reagent model
- `build_reagents.py` — tag a reagent pool against the FG vocabulary; produces `data/reagents_registry.csv` + `data/reagents_index.json`
- `reagents/` — raw reagent pool files (`.smi`); any file placed here is auto-discovered as a selectable reagent set in the UI
- `templates/index.html` — UI
- `run_webapp.sh`, `ts-gnina-webapp.service`, `requirements-webapp.txt`, `TS_WEBAPP_README.md`

**Thompson Sampling engine (upstream)**
- `thompson_sampling.py` — the `ThompsonSampler` class
- `reagent.py` — `Reagent` prior construction/update
- `disallow_tracker.py` — tracks already-sampled products
- `evaluators.py` — scoring functions (FP / MW / ROCS / Fred / DB / ML … and now GNINA via `gnina_evaluator`)
- `baseline.py` — brute-force / random comparisons
- `ts_main.py` — CLI entry point

---

## Command-line use (JSON-driven, upstream engine)

The original CLI still works for the built-in evaluators:

```bash
conda activate ts_gnina
python ts_main.py examples/amide_fp_sim.json
python ts_main.py examples/quinazoline_fp_sim.json
```

For GNINA docking, prefer the web app, or drive `GninaEvaluator` + `RouteSampler` directly
from Python (see the example in [`TS_WEBAPP_README.md`](TS_WEBAPP_README.md#programmatic-use-no-web-app)).

### JSON parameters (`ts_main.py`)

Required: `evaluator_class_name` (e.g. `FPEvaluator`, `MWEvaluator`, `ROCSEvaluator`),
`evaluator_arg`, `reaction_smarts`, `reagent_file_list` (one SMILES file per component),
`num_ts_iterations` (~100–2000), `num_warmup_trials` (3 for 2-component, ~10 for 3+),
`ts_mode` (`maximize` / `minimize`).

Optional: `results_filename`, `log_filename`, `known_std` (std of the score distribution,
scaled to your objective), `minimum_uncertainty` (prior uncertainty; higher → more
exploration, default 0.1).

---

## Credits & license

Thompson Sampling engine © 2023 Patrick Walters ([PatWalters/TS](https://github.com/PatWalters/TS)),
MIT-licensed. GNINA docking integration, filtering, route sampler, and web app added on top.
Released under the MIT License — see [`LICENSE`](LICENSE).
