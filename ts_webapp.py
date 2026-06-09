#!/usr/bin/env python
"""
Thompson Sampling + GNINA docking web app.

A FastAPI front-end that runs the Thompson Sampling search (this repo) over a
linear, multi-step reaction route, scoring each enumerated *final* product by
GNINA docking. Enumerated products are filtered for PAINS / REOS structural
alerts and optional MW / logP ranges before docking (hard filters), reusing the
ligprepper SMARTS sets and the GNINA web-app ligand-prep pipeline.

Run:
    conda activate ts_gnina      # or use run_webapp.sh
    python ts_webapp.py
Then open http://localhost:5011
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

from gnina_evaluator import GninaEvaluator, MolFilters, DockingCancelled, GNINA_PATH, DOCK_CPU
from route_sampler import RouteSampler
from ts_utils import read_reagents

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text()
PORT = int(os.environ.get("TS_WEBAPP_PORT", "5014"))

with open(BASE_DIR / "reactions.json") as fh:
    CATALOG = json.load(fh)
REAGENT_SETS = CATALOG["reagent_sets"]
REACTIONS = {r["id"]: r for r in CATALOG["reactions"]}

# ---------------------------------------------------------------------------
# Reagent pool (synthon model). One tagged master pool + an inverted index
# (class -> block ids), produced by build_reagents.py. When present it enables
# "Master pool (auto-prune)" mode: instead of picking curated set files, each
# reaction component draws live from the pool by the functional-group classes it
# accepts. The curated-set path is unchanged; pool mode is purely additive.
# ---------------------------------------------------------------------------
REGISTRY_CSV = BASE_DIR / "data" / "reagents_registry.csv"
INDEX_JSON = BASE_DIR / "data" / "reagents_index.json"
POOL: Dict[str, dict] = {}            # block id -> {"smiles", "name", "conflict"}
POOL_BY_CLASS: Dict[str, list] = {}   # fg class -> [block ids]
POOL_META: dict = {}


def _load_pool() -> None:
    """Load the tagged registry + inverted index if present (best-effort)."""
    global POOL, POOL_BY_CLASS, POOL_META
    POOL, POOL_BY_CLASS, POOL_META = {}, {}, {}
    if not (REGISTRY_CSV.is_file() and INDEX_JSON.is_file()):
        return
    try:
        df = pd.read_csv(REGISTRY_CSV, dtype=str).fillna("")
        for r in df.itertuples(index=False):
            POOL[r.id] = {"smiles": r.SMILES, "name": r.name or r.id,
                          "conflict": str(r.conflict) in ("1", "1.0", "True")}
        idx = json.loads(INDEX_JSON.read_text())
        POOL_BY_CLASS = idx.get("by_class", {})
        POOL_META = {"n_reagents": idx.get("n_reagents"),
                     "counts": idx.get("counts", {}),
                     "vocab_version": idx.get("vocab_version")}
    except Exception:
        logging.getLogger("ts_webapp").exception("Failed to load reagent pool")
        POOL, POOL_BY_CLASS, POOL_META = {}, {}, {}


_load_pool()


def _pool_candidates(accepts: List[str]) -> List[str]:
    """Block ids matching any of the accepted fg classes (deduped, order-stable)."""
    ids: List[str] = []
    seen = set()
    for cls in accepts:
        for i in POOL_BY_CLASS.get(cls, ()):
            if i not in seen:
                seen.add(i)
                ids.append(i)
    return ids


# Docking is GPU- and disk-cwd-bound; run one TS job at a time.
_JOB_LOCK = threading.Lock()

app = FastAPI(title="Thompson Sampling + GNINA")


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    dir: Path
    status: str = "queued"            # queued | running | done | error | cancelled
    lines: List[str] = field(default_factory=list)
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    evaluator: Optional[GninaEvaluator] = None
    n_results: int = 0
    started: float = field(default_factory=time.time)
    # Search budget, set once the route is known, so the UI can draw a real
    # progress bar / ETA against the evaluator's running evaluation count.
    budget_warmup: int = 0
    budget_search: int = 0
    budget_total: int = 0
    # CNN re-dock sub-run (refine the Vina top-X on the GPU). Tracked separately
    # from the main run so a finished job can be refined without re-running it.
    redock_thread: Optional[threading.Thread] = None
    redock_evaluator: Optional[GninaEvaluator] = None
    redock_cancel: threading.Event = field(default_factory=threading.Event)
    redock_done: int = 0
    redock_total: int = 0
    redock_status: str = ""           # "" | running | done | error | cancelled
    _log_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def log_path(self) -> Path:
        return self.dir / "run.log"

    def _append(self, line: str) -> None:
        """Append a console line to the in-memory buffer and the run.log file.

        Thread-safe: progress callbacks fire from parallel docking threads.
        """
        with self._log_lock:
            self.lines.append(line)
            try:
                with open(self.log_path, "a") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass

    def log(self, msg: str) -> None:
        self._append(f"[{time.strftime('%H:%M:%S')}] {msg}")


JOBS: Dict[str, Job] = {}


class _JobLogHandler(logging.Handler):
    """Mirror TS logger records into a job's line buffer."""

    def __init__(self, job: Job):
        super().__init__()
        self.job = job

    def emit(self, record):
        try:
            self.job._append(f"[{time.strftime('%H:%M:%S')}] {record.getMessage()}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Config parsing / validation
# ---------------------------------------------------------------------------
def _range_or_none(val) -> Optional[tuple]:
    if not val:
        return None
    lo, hi = val
    lo = None if lo in (None, "") else float(lo)
    hi = None if hi in (None, "") else float(hi)
    if lo is None and hi is None:
        return None
    return (lo, hi)


def _resolve_set(set_id: str) -> str:
    if set_id not in REAGENT_SETS:
        raise HTTPException(400, f"Unknown reagent set: {set_id}")
    path = BASE_DIR / REAGENT_SETS[set_id]["file"]
    if not path.is_file():
        raise HTTPException(400, f"Reagent file missing: {path}")
    return str(path)


def _write_meta(job_dir: Path, data: dict) -> None:
    """Persist a job's metadata (config + runtime status) to job.json."""
    try:
        (job_dir / "job.json").write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


def _write_convergence(job_dir: Path, evaluator, score_field: str) -> None:
    """Persist the best-score-so-far curve so past runs can redraw it."""
    try:
        pts = evaluator.convergence()
        (job_dir / "convergence.json").write_text(json.dumps({
            "score_field": score_field,
            "higher_better": evaluator.higher_is_better,
            "docked": evaluator.stats()["docked"],
            "best": evaluator.stats()["best_score"],
            "points": [{"dock": d, "best": b} for d, b in pts],
        }))
    except Exception:
        pass


def _read_meta(job_dir: Path) -> dict:
    p = job_dir / "job.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _job_dir(job_id: str) -> Optional[Path]:
    """Resolve a job's directory from the live registry, falling back to disk so
    that *past* runs (not in the in-memory JOBS dict) are still viewable.
    Returns None if it doesn't exist or escapes JOBS_DIR (path-traversal guard)."""
    job = JOBS.get(job_id)
    if job is not None:
        return job.dir
    d = (JOBS_DIR / job_id)
    try:
        if d.is_dir() and d.resolve().parent == JOBS_DIR.resolve():
            return d
    except Exception:
        pass
    return None


def _slugify(name: str) -> str:
    """Turn a user session name into a safe directory/job id (no path traversal)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return slug[:64]


def _build_route(steps_cfg: List[dict]):
    """Return (reagent_file_list, route_steps, human_summary)."""
    if not steps_cfg:
        raise HTTPException(400, "No reaction steps selected")
    reagent_file_list: List[str] = []
    route_steps = []
    summary = []
    for i, step in enumerate(steps_cfg):
        rid = step.get("reaction_id")
        rxn = REACTIONS.get(rid)
        if rxn is None:
            raise HTTPException(400, f"Unknown reaction: {rid}")
        if i == 0 and rxn.get("role") != "start":
            raise HTTPException(400, f"Step 1 must be a 'start' reaction, got '{rid}'")
        if i > 0 and rxn.get("role") != "extend":
            raise HTTPException(400, f"Step {i+1} must be an 'extend' reaction, got '{rid}'")
        chosen = step.get("reagent_sets", [])
        if len(chosen) != len(rxn["components"]):
            raise HTTPException(
                400,
                f"Reaction '{rid}' needs {len(rxn['components'])} reagent set(s), got {len(chosen)}",
            )
        for set_id in chosen:
            reagent_file_list.append(_resolve_set(set_id))
        route_steps.append((rxn["smarts"], len(chosen)))
        summary.append(
            f"Step {i+1}: {rxn['name']} [{', '.join(REAGENT_SETS[s]['label'] for s in chosen)}]"
        )
    return reagent_file_list, route_steps, summary


def _build_route_pool(steps_cfg: List[dict], work_dir: Path):
    """Pool-mode route: prune the master pool per component by the classes each
    reaction component accepts, writing a per-component .smi into ``work_dir`` so
    the sampler reads it like any other reagent file. Returns the same tuple as
    :func:`_build_route`."""
    if not steps_cfg:
        raise HTTPException(400, "No reaction steps selected")
    if not POOL:
        raise HTTPException(
            400, "No reagent pool loaded — run build_reagents.py to create "
                 "data/reagents_registry.csv + reagents_index.json")
    reagent_file_list: List[str] = []
    route_steps = []
    summary = []
    for i, step in enumerate(steps_cfg):
        rid = step.get("reaction_id")
        rxn = REACTIONS.get(rid)
        if rxn is None:
            raise HTTPException(400, f"Unknown reaction: {rid}")
        if i == 0 and rxn.get("role") != "start":
            raise HTTPException(400, f"Step 1 must be a 'start' reaction, got '{rid}'")
        if i > 0 and rxn.get("role") != "extend":
            raise HTTPException(400, f"Step {i+1} must be an 'extend' reaction, got '{rid}'")
        labels = []
        for j, comp in enumerate(rxn["components"]):
            accepts = comp.get("accepts", [])
            ids = _pool_candidates(accepts)
            if not ids:
                raise HTTPException(
                    400, f"No pool reagents match {comp['label']} "
                         f"({'/'.join(accepts) or 'no classes'}) for reaction '{rid}'")
            path = work_dir / f"pool_s{i}_c{j}.smi"
            with open(path, "w") as fh:
                for k in ids:
                    fh.write(f"{POOL[k]['smiles']} {POOL[k]['name']}\n")
            reagent_file_list.append(str(path))
            labels.append(f"{comp['label']} [{len(ids)} from pool]")
        route_steps.append((rxn["smarts"], len(rxn["components"])))
        summary.append(f"Step {i+1}: {rxn['name']} [{', '.join(labels)}]")
    return reagent_file_list, route_steps, summary


# ---------------------------------------------------------------------------
# Background TS run
# ---------------------------------------------------------------------------
def _run_job(job: Job, cfg: dict, reagent_file_list, route_steps, summary):
    handler = _JobLogHandler(job)
    handler.setLevel(logging.INFO)
    ts_loggers = [logging.getLogger(n) for n in ("thompson_sampling", "route_sampler", "ts_main")]
    with _JOB_LOCK:
        for lg in ts_loggers:
            lg.addHandler(handler)
            lg.setLevel(logging.INFO)
        # Persisted metadata (drives the job-history list and one-click rerun).
        meta = {
            "id": job.id,
            "session_name": cfg.get("session_name"),
            "status": "running",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": cfg,
            "summary": summary,
        }

        def persist():
            _write_meta(job.dir, meta)

        try:
            job.status = "running"
            persist()
            for line in summary:
                job.log(line)

            gnina = cfg["gnina"]
            ts = cfg["ts"]
            fcfg = cfg["filters"]
            mode = "minimize" if not _higher_better(gnina["score_field"]) else "maximize"

            # Random seed: blank -> pick one at random, but always log it so the
            # run can be reproduced. Seeds the global RNGs used by warm-up/search
            # plus the per-dock gnina seed. Note: full reproducibility also needs
            # the same concurrency, since it sets gnina's --cpu.
            seed_cfg = ts.get("seed")
            seed = random.randrange(1, 2**31 - 1) if seed_cfg in (None, "") else int(seed_cfg)
            random.seed(seed)
            np.random.seed(seed)
            job.log(f"Random seed: {seed}")

            job.log(f"Score field: {gnina['score_field']}  ->  TS mode: {mode}")
            job.log("Loading filters (PAINS/REOS/property) ...")
            filters = MolFilters(
                use_pains=fcfg.get("pains", False),
                use_reos=fcfg.get("reos", False),
                mw_range=_range_or_none(fcfg.get("mw")),
                logp_range=_range_or_none(fcfg.get("logp")),
            )
            job.log(
                f"Filters active={filters.active} "
                f"(PAINS {len(filters.pains_patterns)}, REOS {len(filters.reos_rules)}, "
                f"MW {filters.mw_range}, logP {filters.logp_range})"
            )

            # Concurrency: dock several molecules at once, splitting the CPU
            # budget across them. gnina parallelises only ~exhaustiveness ways,
            # so the throughput-maximising default runs DOCK_CPU // exhaustiveness
            # docks in parallel at exhaustiveness CPUs each (full machine, no
            # wasted cores). Override via the "concurrency" config field or the
            # TS_CONCURRENCY env var.
            exhaustiveness = int(gnina.get("exhaustiveness", 8))
            default_conc = max(1, DOCK_CPU // max(1, exhaustiveness))
            req_conc = ts.get("concurrency") or os.environ.get("TS_CONCURRENCY")
            concurrency = default_conc if not req_conc else max(1, int(req_conc))
            cpu_per_dock = max(1, DOCK_CPU // concurrency)
            job.log(
                f"Concurrency: {concurrency} parallel dock(s) x {cpu_per_dock} CPU each "
                f"(budget {DOCK_CPU} CPU, exhaustiveness {exhaustiveness})"
            )
            meta.update(seed=seed, concurrency=concurrency, cpu_per_dock=cpu_per_dock)
            persist()

            eval_dict = {
                "receptor_path": str(job.dir / "receptor.pdb"),
                "score_field": gnina["score_field"],
                "cnn_scoring": gnina.get("cnn_scoring", "none"),
                "exhaustiveness": exhaustiveness,
                "num_modes": gnina.get("num_modes", 9),
                "autobox_add": gnina.get("autobox_add", 4.0),
                "ph": gnina.get("ph", 7.4),
                "gpu_id": cfg.get("gpu_id", 0),
                "cpu": cpu_per_dock,
                "seed": seed,
                "work_dir": str(job.dir / "dock"),
                "filters": filters,
                "cancel_event": job.cancel_event,
                "progress_callback": job.log,
            }
            bs = cfg["binding_site"]
            if bs["mode"] == "reference":
                eval_dict["reference_path"] = str(job.dir / "reference.sdf")
            else:
                eval_dict["center"] = tuple(bs["center"])
                eval_dict["box_size"] = bs.get("box_size", 16.0)

            evaluator = GninaEvaluator(eval_dict)
            job.evaluator = evaluator

            sampler = RouteSampler(mode=mode)
            sampler.set_hide_progress(True)
            sampler.set_concurrency(concurrency)
            sampler.set_seed(seed)
            sampler.set_evaluator(evaluator)
            sampler.read_reagents(reagent_file_list=reagent_file_list, num_to_select=None)
            sampler.set_route(route_steps)
            n_components = sum(n for _s, n in route_steps)
            if n_components != len(sampler.reagent_lists):
                raise RuntimeError("Internal: reagent component count mismatch")
            job.log(f"{sampler.get_num_prods():.2e} possible final products across the route")

            n_warm = int(ts.get("num_warmup_trials", 3))
            method = (ts.get("method") or "ts").lower()

            # Optional plateau auto-stop: end the search once the best score has
            # not improved for this many consecutive docks. Blank / 0 disables it.
            pat_cfg = ts.get("patience")
            patience = int(pat_cfg) if pat_cfg not in (None, "") and int(pat_cfg) > 0 else None
            if patience:
                job.log(f"Auto-stop enabled: stop if no score improvement in {patience} docks")

            def _no_warmup_scores() -> RuntimeError:
                st = evaluator.stats()
                return RuntimeError(
                    "Warm-up produced no scorable products — nothing to build a prior from. "
                    f"Of the products tried: {sum(st['rejections'].values())} filtered out before docking "
                    f"({st['rejections'] or 'none'}), {st['prep_failures']} ligand-prep failures, "
                    f"{st['dock_failures']} docking failures. "
                    "Loosen the MW/logP ranges or PAINS/REOS filters, or check that the "
                    "reaction and receptor/binding site are correct."
                )

            if method == "rws":
                # Roulette Wheel Sampling + thermal cycling (Zhao et al. 2025).
                # Budget is an absolute product count, shared with the TS search
                # field, so a TS vs RWS run on the same budget is comparable.
                num_targets = int(ts.get("num_ts_iterations", 100))
                mcpc = int(ts.get("min_cpds_per_core", 50))
                stop = int(ts.get("stop", 6000))
                est_warm = sum(len(rl) for rl in sampler.reagent_lists) * n_warm
                job.budget_warmup, job.budget_search = est_warm, num_targets
                job.budget_total = est_warm + num_targets
                job.log("Selection: Roulette Wheel Sampling + thermal cycling (Zhao 2025)")
                job.log(f"Warm-up: ~{est_warm} docking calls; then RWS search budget "
                        f"{num_targets} products (min {mcpc}/core, resample-stop {stop})")
                warmup_results = sampler.warm_up_rws(num_warmup_trials=n_warm)
                if not warmup_results:
                    raise _no_warmup_scores()
                search_results = sampler.search_rws(
                    num_targets=num_targets, min_cpds_per_core=mcpc, stop=stop, patience=patience)
            else:
                est = sum(len(rl) for rl in sampler.reagent_lists) * n_warm
                n_cycles = int(ts.get("num_ts_iterations", 100))
                job.budget_warmup, job.budget_search = est, n_cycles
                job.budget_total = est + n_cycles
                job.log("Selection: standard Thompson Sampling (argmax)")
                job.log(f"Warm-up: ~{est} docking calls (then {ts.get('num_ts_iterations')} search iterations)")
                try:
                    warmup_results = sampler.warm_up(num_warmup_trials=n_warm)
                except ValueError:
                    # warm_up() computes np.min/np.mean over the finite warm-up
                    # scores; if every product was filtered out or failed to dock
                    # that array is empty and numpy raises. Turn it into guidance.
                    raise _no_warmup_scores()
                if not warmup_results:
                    raise _no_warmup_scores()
                try:
                    search_results = sampler.search(num_cycles=n_cycles, patience=patience)
                except ValueError as e:
                    # nanargmin/nanargmax raise "All-NaN slice encountered" once the
                    # disallow tracker has exhausted a (typically very small) library.
                    job.log(f"Search ended early — library effectively exhausted ({e}). "
                            f"Reporting results gathered so far.")
                    search_results = []

            # Warm-up products are docked too; include them so nothing is lost.
            out_list = warmup_results + search_results

            # Results
            out_df = pd.DataFrame(out_list, columns=["score", "SMILES", "Name"])
            out_df = out_df.dropna(subset=["score"])
            ascending = (mode == "minimize")
            out_df = out_df.sort_values("score", ascending=ascending).drop_duplicates(subset="SMILES")
            results_csv = job.dir / "results.csv"
            out_df.to_csv(results_csv, index=False)
            job.n_results = len(out_df)

            poses_sdf = job.dir / "poses.sdf"
            n_poses = evaluator.write_top_poses(str(poses_sdf), n=200)
            _write_convergence(job.dir, evaluator, gnina["score_field"])

            stats = evaluator.stats()
            job.log(f"Done. Unique scored: {stats['unique_scored']} | docked: {stats['docked']}")
            job.log(f"Filtered out (pre-dock): {sum(stats['rejections'].values())} {stats['rejections']}")
            job.log(f"Prep failures: {stats['prep_failures']} | dock failures: {stats['dock_failures']}")
            job.log(f"Results: {job.n_results} unique molecules, {n_poses} poses written")
            top_meta = None
            if not out_df.empty:
                top = out_df.iloc[0]
                job.log(f"Top hit: {top['SMILES']}  score={top['score']:.3f}  ({top['Name']})")
                top_meta = {"smiles": str(top["SMILES"]), "score": float(top["score"]), "name": str(top["Name"])}
            job.status = "done"
            meta.update(status="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                        n_results=job.n_results, n_poses=n_poses, top=top_meta)
            persist()

        except DockingCancelled:
            job.status = "cancelled"
            job.log("Run cancelled by user.")
            if job.evaluator is not None:
                _write_convergence(job.dir, job.evaluator, cfg["gnina"]["score_field"])
            meta.update(status="cancelled", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            persist()
        except Exception as e:  # noqa
            job.status = "error"
            job.error = str(e)
            job.log(f"ERROR: {e}")
            logging.getLogger("ts_webapp").exception("Job failed")
            meta.update(status="error", error=str(e), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            persist()
        finally:
            for lg in ts_loggers:
                lg.removeHandler(handler)


def _higher_better(score_field: str) -> bool:
    from gnina_evaluator import score_field_is_higher_better
    return score_field_is_higher_better(score_field)


# ---------------------------------------------------------------------------
# CNN re-dock (post-run refinement of the Vina top-X on the GPU)
# ---------------------------------------------------------------------------
def _redock_job(job: Job, cfg: dict, rows: List[tuple], params: dict):
    """Re-dock the top-X products of a finished run with a CNN scoring mode.

    The cheap default run uses Vina (CPU) docking; CNN scoring is more accurate
    but GPU-bound, so we refine only the shortlist here. Results land in
    ``results_cnn.csv`` / ``poses_cnn.sdf`` next to the original outputs and are
    surfaced through the gallery's "CNN re-dock" source toggle."""
    with _JOB_LOCK:
        score_field = params["score_field"]
        cnn_scoring = params["cnn_scoring"]
        higher = _higher_better(score_field)
        mode = "maximize" if higher else "minimize"
        meta = _read_meta(job.dir)
        rmeta = {
            "status": "running",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_x": len(rows),
            "cnn_scoring": cnn_scoring,
            "score_field": score_field,
        }
        meta["redock"] = rmeta
        _write_meta(job.dir, meta)
        try:
            job.redock_status = "running"
            job.redock_done = 0
            job.redock_total = len(rows)
            gnina = cfg["gnina"]
            job.log(f"CNN re-dock: refining top {len(rows)} products "
                    f"(cnn_scoring={cnn_scoring}, field={score_field}, GPU {cfg.get('gpu_id', 0)})")

            eval_dict = {
                "receptor_path": str(job.dir / "receptor.pdb"),
                "score_field": score_field,
                "cnn_scoring": cnn_scoring,
                "exhaustiveness": int(gnina.get("exhaustiveness", 8)),
                "num_modes": int(gnina.get("num_modes", 9)),
                "autobox_add": gnina.get("autobox_add", 4.0),
                "ph": gnina.get("ph", 7.4),
                "gpu_id": cfg.get("gpu_id", 0),
                "cpu": DOCK_CPU,
                "seed": meta.get("seed", 666),
                "work_dir": str(job.dir / "redock"),
                "filters": None,  # the shortlist already passed the run's filters
                "cancel_event": job.redock_cancel,
                "progress_callback": job.log,
                "progress_every": 1,
            }
            bs = cfg["binding_site"]
            if bs["mode"] == "reference":
                eval_dict["reference_path"] = str(job.dir / "reference.sdf")
            else:
                eval_dict["center"] = tuple(bs["center"])
                eval_dict["box_size"] = bs.get("box_size", 16.0)

            evaluator = GninaEvaluator(eval_dict)
            job.redock_evaluator = evaluator

            for smi, name in rows:
                if job.redock_cancel.is_set():
                    raise DockingCancelled()
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    job.redock_done += 1
                    continue
                mol.SetProp("_Name", str(name))
                evaluator.evaluate(mol)
                job.redock_done += 1

            scored = evaluator.top_scored(len(rows))
            out_df = pd.DataFrame(scored, columns=["score", "SMILES", "Name"])
            out_df = out_df.dropna(subset=["score"]).drop_duplicates(subset="SMILES")
            out_df = out_df.sort_values("score", ascending=not higher)
            out_df.to_csv(job.dir / "results_cnn.csv", index=False)
            n_poses = evaluator.write_top_poses(str(job.dir / "poses_cnn.sdf"), n=200)

            st = evaluator.stats()
            job.redock_status = "done"
            job.log(f"CNN re-dock done. Scored {len(out_df)} | dock failures {st['dock_failures']} "
                    f"| best {score_field}={st['best_score']}")
            if not out_df.empty:
                top = out_df.iloc[0]
                job.log(f"CNN top: {top['SMILES']}  {score_field}={top['score']:.3f}  ({top['Name']})")
            rmeta.update(status="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                         n_results=int(len(out_df)), n_poses=n_poses,
                         best_score=st["best_score"])
        except DockingCancelled:
            job.redock_status = "cancelled"
            job.log("CNN re-dock cancelled by user.")
            rmeta.update(status="cancelled", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:  # noqa
            job.redock_status = "error"
            job.log(f"CNN re-dock ERROR: {e}")
            logging.getLogger("ts_webapp").exception("Re-dock failed")
            rmeta.update(status="error", error=str(e), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        finally:
            meta["redock"] = rmeta
            _write_meta(job.dir, meta)
            job.redock_cancel.clear()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    gnina_ok = os.path.exists(GNINA_PATH) or shutil.which("gnina") is not None
    pool_info = {"available": bool(POOL), "size": POOL_META.get("n_reagents"),
                 "counts": POOL_META.get("counts", {})}
    html = (
        INDEX_HTML
        .replace("__CATALOG_JSON__", json.dumps(CATALOG))
        .replace("__GNINA_OK__", "true" if gnina_ok else "false")
        .replace("__POOL_JSON__", json.dumps(pool_info))
    )
    return HTMLResponse(html)


@app.post("/run")
async def run(
    config: str = Form(...),
    receptor: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
):
    cfg = json.loads(config)
    source = cfg.get("reagent_source", "sets")
    # Curated-set routes validate before any dir is created; pool routes need the
    # job dir (they write per-component .smi into it), so they're built below.
    if source != "pool":
        reagent_file_list, route_steps, summary = _build_route(cfg.get("steps", []))

    # Optional human-readable session name. Re-running the same name overwrites
    # that session's previous results; a blank name falls back to a random id
    # (which never collides).
    session_name = _slugify(cfg.get("session_name") or "")
    job_id = session_name or uuid.uuid4().hex[:12]
    existing = JOBS.get(job_id)
    if existing is not None and existing.status in ("queued", "running"):
        raise HTTPException(409, f"A session named '{job_id}' is already running")
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)  # overwrite previous results
    job_dir.mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, dir=job_dir)
    JOBS[job_id] = job

    if source == "pool":
        try:
            reagent_file_list, route_steps, summary = _build_route_pool(cfg.get("steps", []), job_dir)
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            JOBS.pop(job_id, None)
            raise

    # Save uploads
    with open(job_dir / "receptor.pdb", "wb") as f:
        f.write(await receptor.read())
    bs = cfg.get("binding_site", {})
    if bs.get("mode") == "reference":
        if reference is None:
            raise HTTPException(400, "Reference ligand SDF required for autobox mode")
        with open(job_dir / "reference.sdf", "wb") as f:
            f.write(await reference.read())

    job.thread = threading.Thread(
        target=_run_job, args=(job, cfg, reagent_file_list, route_steps, summary), daemon=True
    )
    job.thread.start()
    return {"job_id": job_id}


@app.get("/jobs")
async def list_jobs():
    """List past + current runs (newest first) for the history panel."""
    items = []
    for d in sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta = _read_meta(d)
        live = JOBS.get(d.name)
        has_results = (d / "results.csv").is_file()
        status = (live.status if live is not None
                  else meta.get("status") or ("done" if has_results else "unknown"))
        items.append({
            "id": d.name,
            "session_name": meta.get("session_name"),
            "status": status,
            "started": meta.get("started"),
            "finished": meta.get("finished"),
            "n_results": (live.n_results if live is not None else meta.get("n_results")),
            "seed": meta.get("seed"),
            "concurrency": meta.get("concurrency"),
            "top": meta.get("top"),
            "summary": meta.get("summary"),
            "has_results": has_results,
            "has_poses": (d / "poses.sdf").is_file(),
            "has_log": (d / "run.log").is_file(),
            "has_cnn": (d / "results_cnn.csv").is_file(),
            "redock": meta.get("redock"),
            "can_rerun": bool(meta.get("config")) and (d / "receptor.pdb").is_file(),
            "can_redock": has_results and bool(meta.get("config")) and (d / "receptor.pdb").is_file(),
        })
    return {"jobs": items}


@app.post("/jobs/{job_id}/rerun")
async def rerun(job_id: str):
    """Re-run a past job from its stored config + receptor (no re-upload)."""
    src = JOBS_DIR / job_id
    meta = _read_meta(src)
    cfg = meta.get("config")
    if not cfg:
        raise HTTPException(400, "No stored config for this run; cannot rerun")
    if not (src / "receptor.pdb").is_file():
        raise HTTPException(400, "Stored receptor is missing; cannot rerun")
    live = JOBS.get(job_id)
    if live is not None and live.status in ("queued", "running"):
        raise HTTPException(409, f"'{job_id}' is already running")

    source = cfg.get("reagent_source", "sets")
    if source != "pool":
        reagent_file_list, route_steps, summary = _build_route(cfg.get("steps", []))

    # Preserve the uploaded receptor/reference across the in-place overwrite.
    has_ref = (src / "reference.sdf").is_file()
    tmp = Path(tempfile.mkdtemp(prefix="ts_rerun_"))
    shutil.copy(src / "receptor.pdb", tmp / "receptor.pdb")
    if has_ref:
        shutil.copy(src / "reference.sdf", tmp / "reference.sdf")
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp / "receptor.pdb", src / "receptor.pdb")
    if has_ref:
        shutil.copy(tmp / "reference.sdf", src / "reference.sdf")
    shutil.rmtree(tmp, ignore_errors=True)

    if source == "pool":  # pool routes write per-component .smi into the job dir
        reagent_file_list, route_steps, summary = _build_route_pool(cfg.get("steps", []), src)

    job = Job(id=job_id, dir=src)
    JOBS[job_id] = job
    job.thread = threading.Thread(
        target=_run_job, args=(job, cfg, reagent_file_list, route_steps, summary), daemon=True
    )
    job.thread.start()
    return {"job_id": job_id}


@app.post("/candidates")
async def candidates(config: str = Form(...)):
    """Pool-mode preview: for the chosen route, how many master-pool reagents
    each component accepts, and the resulting combinatorial library size. Pure
    lookups against the inverted index — fast enough to drive the route UI."""
    cfg = json.loads(config)
    if not POOL:
        return {"pool_available": False}
    comps = []
    lib = 1
    for i, step in enumerate(cfg.get("steps", [])):
        rxn = REACTIONS.get(step.get("reaction_id"))
        if rxn is None:
            raise HTTPException(400, f"Unknown reaction: {step.get('reaction_id')}")
        for comp in rxn["components"]:
            accepts = comp.get("accepts", [])
            n = len(_pool_candidates(accepts))
            lib *= n
            comps.append({"step": i + 1, "label": comp["label"],
                          "accepts": accepts, "count": n})
    return {"pool_available": True, "pool_size": POOL_META.get("n_reagents"),
            "components": comps, "library_size": lib}


@app.post("/preflight")
async def preflight(config: str = Form(...)):
    """Sample random products and report filter pass-rate + MW/logP spread,
    so a too-tight filter is caught before committing to a full run."""
    cfg = json.loads(config)
    pool_tmp = None
    if cfg.get("reagent_source") == "pool":
        pool_tmp = Path(tempfile.mkdtemp(prefix="ts_pf_"))
        reagent_file_list, route_steps, _summary = _build_route_pool(cfg.get("steps", []), pool_tmp)
    else:
        reagent_file_list, route_steps, _summary = _build_route(cfg.get("steps", []))
    fcfg = cfg.get("filters", {})
    filters = MolFilters(
        use_pains=fcfg.get("pains", False),
        use_reos=fcfg.get("reos", False),
        mw_range=_range_or_none(fcfg.get("mw")),
        logp_range=_range_or_none(fcfg.get("logp")),
    )
    sampler = RouteSampler(mode="minimize")
    sampler.read_reagents(reagent_file_list=reagent_file_list, num_to_select=None)
    sampler.set_route(route_steps)
    if pool_tmp is not None:  # reagents are in memory now; drop the temp .smi
        shutil.rmtree(pool_tmp, ignore_errors=True)

    n = 300
    built = 0
    reaction_fail = 0
    passed = 0
    rejections: Dict[str, int] = {}
    mws: List[float] = []
    logps: List[float] = []
    for _ in range(n):
        choice = [random.randrange(len(rl)) for rl in sampler.reagent_lists]
        prod_mol, _smi, _name, _sel = sampler._build_product(choice)
        if prod_mol is None:
            reaction_fail += 1
            continue
        built += 1
        mws.append(Descriptors.MolWt(prod_mol))
        logps.append(Crippen.MolLogP(prod_mol))
        reason = filters.reject_reason(prod_mol) if filters.active else None
        if reason is None:
            passed += 1
        else:
            key = reason.split(":")[0].split(" ")[0]
            rejections[key] = rejections.get(key, 0) + 1

    def spread(vals):
        if not vals:
            return None
        a = np.array(vals)
        return {k: round(float(v), 2) for k, v in
                zip(("min", "p10", "median", "p90", "max"),
                    np.percentile(a, [0, 10, 50, 90, 100]))}

    return {
        "sampled": n,
        "built": built,
        "reaction_fail": reaction_fail,
        "passed": passed,
        "pass_rate": round(passed / built, 3) if built else 0.0,
        "filters_active": filters.active,
        "rejections": rejections,
        "mw": spread(mws),
        "logp": spread(logps),
        # Full combinatorial library = product of the per-component reagent
        # counts (upper bound; the sampled built/passed fractions scale it to the
        # effective enumerable+passing size).
        "library_size": int(sampler.get_num_prods()),
    }


@app.post("/extend-options")
async def extend_options(config: str = Form(...)):
    """For a prospective extension step, report how often each extend reaction
    actually chains onto the products of the already-chosen upstream steps.

    Compatibility is reagent-dependent: the upstream product only carries the
    handle the next reaction needs when the chosen reagents happen to be
    difunctional. So rather than guess from the reaction templates, we sample
    real upstream intermediates and try each extend reaction against them, using
    each reaction's default reagent set. Build-only (no docking), so it's fast
    enough to drive the step-2 dropdown."""
    cfg = json.loads(config)
    upstream = cfg.get("steps", [])
    if not upstream:
        raise HTTPException(400, "No upstream steps to extend")
    reagent_file_list, route_steps, _ = _build_route(upstream)
    up = RouteSampler(mode="minimize")
    # Cap reads: a representative sample is plenty and keeps the dropdown snappy.
    up.read_reagents(reagent_file_list=reagent_file_list, num_to_select=400)
    up.set_route(route_steps)

    n = 200
    intermediates = []
    for _ in range(n):
        choice = [random.randrange(len(rl)) for rl in up.reagent_lists]
        mol, _smi, _name, _sel = up._build_product(choice)
        if mol is not None:
            intermediates.append(mol)

    results: Dict[str, dict] = {}
    for r in CATALOG["reactions"]:
        if r.get("role") != "extend":
            continue
        rxn = AllChem.ReactionFromSmarts(r["smarts"])
        # Default reagent set = first listed for each component. The new reagent
        # rarely decides whether the reaction fires (its reactive group is fixed
        # by the set), so this is representative regardless of the final pick.
        new_lists = [read_reagents([_resolve_set(c["sets"][0])], 300)[0]
                     for c in r["components"]]
        built = 0
        for inter in intermediates:
            reactants = [inter] + [random.choice(rl).mol for rl in new_lists]
            try:
                prods = rxn.RunReactants(reactants)
                if prods:
                    Chem.SanitizeMol(prods[0][0])
                    built += 1
            except Exception:
                pass
        denom = len(intermediates)
        results[r["id"]] = {
            "built": built,
            "chain_rate": round(built / denom, 3) if denom else 0.0,
        }
    return {
        "upstream_built": len(intermediates),
        "upstream_sampled": n,
        "reactions": results,
    }


@app.get("/jobs/{job_id}/status")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    # Progress for the live bar / ETA. The evaluator's evaluation counter advances
    # ~one per warm-up product and one per search iteration (cache hits included),
    # so it tracks the budget linearly; the client turns elapsed + fraction into
    # an ETA. Docked/best come from the same evaluator for a one-line summary.
    ev = job.evaluator
    docked = best = None
    evaluations = 0
    if ev is not None:
        st = ev.stats()
        evaluations = st["evaluations"]
        docked = st["docked"]
        best = st["best_score"]
    phase = "warm-up" if (job.budget_warmup and evaluations <= job.budget_warmup) else "search"
    return {
        "status": job.status,
        "n_lines": len(job.lines),
        "n_results": job.n_results,
        "error": job.error,
        "evaluations": evaluations,
        "docked": docked,
        "best_score": best,
        "budget_total": job.budget_total,
        "budget_warmup": job.budget_warmup,
        "phase": phase,
        "elapsed": round(time.time() - job.started, 1),
    }


@app.get("/jobs/{job_id}/stream")
async def stream(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")

    async def gen():
        import asyncio
        sent = 0
        while True:
            while sent < len(job.lines):
                yield f"data: {job.lines[sent]}\n\n"
                sent += 1
            if job.status in ("done", "error", "cancelled") and sent >= len(job.lines):
                yield f"event: end\ndata: {job.status}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/jobs/{job_id}/cancel")
async def cancel(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    job.cancel_event.set()
    return {"status": "cancelling"}


@app.post("/jobs/{job_id}/redock")
async def redock(job_id: str, top_x: int = Form(20),
                 cnn_scoring: str = Form("rescore"), score_field: str = Form("CNNaffinity")):
    """Re-dock a finished run's top-X products with CNN scoring (GPU) for a more
    accurate final ranking. Reuses the stored config (binding site, receptor)."""
    if cnn_scoring not in ("rescore", "all"):
        raise HTTPException(400, "cnn_scoring must be 'rescore' or 'all'")
    if score_field not in ("CNNaffinity", "CNN_VS", "CNNscore"):
        raise HTTPException(400, f"Unsupported CNN score field: {score_field}")
    d = _job_dir(job_id)
    if d is None:
        raise HTTPException(404, "Unknown job")
    meta = _read_meta(d)
    cfg = meta.get("config")
    if not cfg:
        raise HTTPException(400, "No stored config for this run; cannot re-dock")
    if not (d / "receptor.pdb").is_file():
        raise HTTPException(400, "Stored receptor is missing; cannot re-dock")
    csv_path = d / "results.csv"
    if not csv_path.is_file():
        raise HTTPException(400, "No results.csv to re-dock")
    if _JOB_LOCK.locked():
        raise HTTPException(409, "Another run is in progress; try again when it finishes")

    # Resurrect a Job handle if this is a past run not in the live registry, so
    # progress + the live gallery work the same as for a just-finished run.
    job = JOBS.get(job_id)
    if job is None:
        job = Job(id=job_id, dir=d, status="done")
        JOBS[job_id] = job
    if job.redock_status == "running":
        raise HTTPException(409, "A CNN re-dock is already running for this job")

    top_x = max(1, min(int(top_x), 200))
    df = pd.read_csv(csv_path)  # already sorted best-first
    rows = [(r.SMILES, r.Name) for r in df.head(top_x).itertuples(index=False)]
    if not rows:
        raise HTTPException(400, "No products to re-dock")

    job.redock_cancel.clear()
    job.redock_thread = threading.Thread(
        target=_redock_job,
        args=(job, cfg, rows, {"cnn_scoring": cnn_scoring, "score_field": score_field}),
        daemon=True,
    )
    job.redock_thread.start()
    return {"job_id": job_id, "top_x": len(rows)}


@app.get("/jobs/{job_id}/redock/status")
async def redock_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        # Past run reloaded after a restart: report from disk.
        d = _job_dir(job_id)
        meta = _read_meta(d) if d is not None else {}
        r = meta.get("redock") or {}
        return {"status": r.get("status", ""), "done": 0,
                "total": r.get("top_x", 0), "best_score": r.get("best_score")}
    best = job.redock_evaluator.stats()["best_score"] if job.redock_evaluator is not None else None
    return {
        "status": job.redock_status,
        "done": job.redock_done,
        "total": job.redock_total,
        "best_score": best,
    }


@app.post("/jobs/{job_id}/redock/cancel")
async def redock_cancel(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job.redock_status != "running":
        raise HTTPException(404, "No running re-dock for this job")
    job.redock_cancel.set()
    return {"status": "cancelling"}


def _mol_svg(smiles: str, width: int = 230, height: int = 180) -> str:
    """Render a SMILES to an inline SVG (XML declaration stripped)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    i = svg.find("<svg")
    return svg[i:] if i != -1 else svg


def _top_items(rows) -> list:
    """Render ``(score, smiles, name)`` rows (best-first) into gallery items."""
    items = []
    for rank, (score, smiles, name) in enumerate(rows, start=1):
        items.append({
            "rank": rank,
            "score": round(float(score), 3),
            "smiles": str(smiles),
            "name": str(name),
            "svg": _mol_svg(str(smiles)),
        })
    return items


@app.get("/jobs/{job_id}/top")
async def top(job_id: str, n: int = 12, source: str = "vina"):
    """Top-N ranked products with structure SVG, score and reagent combination.

    ``source`` selects the original Vina run (``vina``, ``results.csv``) or the
    CNN re-dock refinement (``cnn``, ``results_cnn.csv``). While the relevant
    pass is in progress the standings are read live from its evaluator; once
    finished (or for past runs) they come from the persisted CSV."""
    n = max(1, min(int(n), 60))
    is_cnn = source == "cnn"

    # Live view: a running pass's evaluator holds every score gathered so far.
    live = JOBS.get(job_id)
    if is_cnn:
        if live is not None and live.redock_status == "running" and live.redock_evaluator is not None:
            rows = live.redock_evaluator.top_scored(n)
            total = live.redock_evaluator.stats()["unique_scored"]
            return {"ready": bool(rows), "live": True, "items": _top_items(rows), "total": total}
    elif live is not None and live.status == "running" and live.evaluator is not None:
        rows = live.evaluator.top_scored(n)
        total = live.evaluator.stats()["unique_scored"]
        return {"ready": bool(rows), "live": True, "items": _top_items(rows), "total": total}

    d = _job_dir(job_id)
    if d is None:
        raise HTTPException(404, "Unknown job")
    csv_path = d / ("results_cnn.csv" if is_cnn else "results.csv")
    if not csv_path.is_file():
        return {"ready": False, "live": False, "items": []}
    try:
        df = pd.read_csv(csv_path)  # already sorted best-first and deduped
    except Exception:
        return {"ready": False, "live": False, "items": []}
    rows = [(row.score, row.SMILES, row.Name) for row in df.head(n).itertuples(index=False)]
    return {"ready": True, "live": False, "items": _top_items(rows), "total": int(len(df))}


@app.get("/jobs/{job_id}/convergence")
async def convergence(job_id: str):
    """Best-score-so-far vs docks, for the convergence chart. Served live from a
    running job's evaluator, else from the persisted convergence.json."""
    live = JOBS.get(job_id)
    if live is not None and live.status == "running" and live.evaluator is not None:
        ev = live.evaluator
        pts = ev.convergence()
        st = ev.stats()
        return {
            "ready": bool(pts), "live": True,
            "score_field": ev.score_field, "higher_better": ev.higher_is_better,
            "docked": st["docked"], "best": st["best_score"],
            "points": [{"dock": d, "best": b} for d, b in pts],
        }
    d = _job_dir(job_id)
    if d is None:
        raise HTTPException(404, "Unknown job")
    p = d / "convergence.json"
    if not p.is_file():
        return {"ready": False, "live": False, "points": []}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {"ready": False, "live": False, "points": []}
    data["ready"] = bool(data.get("points"))
    data["live"] = False
    return data


@app.get("/jobs/{job_id}/download/{name}")
async def download(job_id: str, name: str):
    d = _job_dir(job_id)
    if d is None:
        raise HTTPException(404, "Unknown job")
    if name not in ("results.csv", "poses.sdf", "run.log", "results_cnn.csv", "poses_cnn.sdf"):
        raise HTTPException(400, "Invalid file")
    path = d / name
    if not path.is_file():
        raise HTTPException(404, "Not ready")
    return FileResponse(str(path), filename=f"ts_gnina_{job_id}_{name}")


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a run's directory (prune clutter). Refuses a running job."""
    live = JOBS.get(job_id)
    if live is not None and live.status in ("queued", "running"):
        raise HTTPException(409, "Cannot delete a running job; cancel it first")
    d = _job_dir(job_id)
    if d is None:
        raise HTTPException(404, "Unknown job")
    shutil.rmtree(d, ignore_errors=True)
    JOBS.pop(job_id, None)
    return {"deleted": job_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
