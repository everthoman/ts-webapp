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
import shutil
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

from gnina_evaluator import GninaEvaluator, MolFilters, DockingCancelled, GNINA_PATH, DOCK_CPU
from route_sampler import RouteSampler

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text()
PORT = int(os.environ.get("TS_WEBAPP_PORT", "5014"))

with open(BASE_DIR / "reactions.json") as fh:
    CATALOG = json.load(fh)
REAGENT_SETS = CATALOG["reagent_sets"]
REACTIONS = {r["id"]: r for r in CATALOG["reactions"]}

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
        try:
            job.status = "running"
            for line in summary:
                job.log(line)

            gnina = cfg["gnina"]
            ts = cfg["ts"]
            fcfg = cfg["filters"]
            mode = "minimize" if not _higher_better(gnina["score_field"]) else "maximize"

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
            sampler.set_evaluator(evaluator)
            sampler.read_reagents(reagent_file_list=reagent_file_list, num_to_select=None)
            sampler.set_route(route_steps)
            n_components = sum(n for _s, n in route_steps)
            if n_components != len(sampler.reagent_lists):
                raise RuntimeError("Internal: reagent component count mismatch")
            job.log(f"{sampler.get_num_prods():.2e} possible final products across the route")

            n_warm = int(ts.get("num_warmup_trials", 3))
            est = sum(len(rl) for rl in sampler.reagent_lists) * n_warm
            job.log(f"Warm-up: ~{est} docking calls (then {ts.get('num_ts_iterations')} search iterations)")

            try:
                warmup_results = sampler.warm_up(num_warmup_trials=n_warm)
            except ValueError:
                # warm_up() computes np.min/np.mean over the finite warm-up
                # scores; if every product was filtered out or failed to dock
                # that array is empty and numpy raises. Turn it into guidance.
                st = evaluator.stats()
                rej = sum(st["rejections"].values())
                raise RuntimeError(
                    "Warm-up produced no scorable products — nothing to build a prior from. "
                    f"Of the products tried: {rej} filtered out before docking "
                    f"({st['rejections'] or 'none'}), {st['prep_failures']} ligand-prep failures, "
                    f"{st['dock_failures']} docking failures. "
                    "Loosen the MW/logP ranges or PAINS/REOS filters, or check that the "
                    "reaction and receptor/binding site are correct."
                )
            if not warmup_results:
                st = evaluator.stats()
                raise RuntimeError(
                    "Warm-up produced no scorable products. "
                    f"Filtered out: {sum(st['rejections'].values())} {st['rejections'] or ''}; "
                    f"prep failures: {st['prep_failures']}; dock failures: {st['dock_failures']}. "
                    "Loosen the filters or check the reaction/receptor."
                )
            try:
                search_results = sampler.search(num_cycles=int(ts.get("num_ts_iterations", 100)))
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

            stats = evaluator.stats()
            job.log(f"Done. Unique scored: {stats['unique_scored']} | docked: {stats['docked']}")
            job.log(f"Filtered out (pre-dock): {sum(stats['rejections'].values())} {stats['rejections']}")
            job.log(f"Prep failures: {stats['prep_failures']} | dock failures: {stats['dock_failures']}")
            job.log(f"Results: {job.n_results} unique molecules, {n_poses} poses written")
            if not out_df.empty:
                top = out_df.iloc[0]
                job.log(f"Top hit: {top['SMILES']}  score={top['score']:.3f}  ({top['Name']})")
            job.status = "done"

        except DockingCancelled:
            job.status = "cancelled"
            job.log("Run cancelled by user.")
        except Exception as e:  # noqa
            job.status = "error"
            job.error = str(e)
            job.log(f"ERROR: {e}")
            logging.getLogger("ts_webapp").exception("Job failed")
        finally:
            for lg in ts_loggers:
                lg.removeHandler(handler)


def _higher_better(score_field: str) -> bool:
    from gnina_evaluator import score_field_is_higher_better
    return score_field_is_higher_better(score_field)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    gnina_ok = os.path.exists(GNINA_PATH) or shutil.which("gnina") is not None
    html = (
        INDEX_HTML
        .replace("__CATALOG_JSON__", json.dumps(CATALOG))
        .replace("__GNINA_OK__", "true" if gnina_ok else "false")
    )
    return HTMLResponse(html)


@app.post("/run")
async def run(
    config: str = Form(...),
    receptor: UploadFile = File(...),
    reference: Optional[UploadFile] = File(None),
):
    cfg = json.loads(config)
    reagent_file_list, route_steps, summary = _build_route(cfg.get("steps", []))

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, dir=job_dir)
    JOBS[job_id] = job

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


@app.get("/jobs/{job_id}/status")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return {
        "status": job.status,
        "n_lines": len(job.lines),
        "n_results": job.n_results,
        "error": job.error,
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


@app.get("/jobs/{job_id}/download/{name}")
async def download(job_id: str, name: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    if name not in ("results.csv", "poses.sdf", "run.log"):
        raise HTTPException(400, "Invalid file")
    path = job.dir / name
    if not path.is_file():
        raise HTTPException(404, "Not ready")
    return FileResponse(str(path), filename=f"ts_gnina_{job_id}_{name}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
