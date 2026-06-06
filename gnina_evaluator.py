"""
GNINA docking evaluator for Thompson Sampling.

This module provides a TS ``Evaluator`` that scores a molecule by docking it
with GNINA, reusing the ligand-preparation pipeline of the GNINA web app
(OpenBabel protonation at a target pH followed by RDKit ETKDGv3 embedding and
MMFF94s minimisation) and the PAINS/REOS structural-alert definitions shipped
with ``ligprepper``.

Design notes
------------
* Every enumerated product is first run through :class:`MolFilters` (PAINS,
  REOS, MW range, logP range). A molecule that fails any enabled filter is
  *rejected before docking* by returning ``np.nan``. The Thompson Sampling
  engine already treats ``np.nan`` scores as "skip" (see the ``np.isfinite``
  checks in ``thompson_sampling.py``): the reagent priors are not updated and
  the product is not counted, which is exactly the desired behaviour for a
  hard filter.
* Scores are cached by canonical SMILES. Thompson Sampling re-samples the same
  product many times, and docking is by far the rate-limiting step, so caching
  avoids redundant GNINA calls.
* The best docked pose for every successfully scored molecule is retained so
  the web app can export an SDF of the top hits.

The class is intentionally self-contained (it only imports the ``Evaluator``
ABC from ``evaluators``) so that importing it never pulls in the optional
OpenEye / joblib dependencies.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors

from evaluators import Evaluator

# ---------------------------------------------------------------------------
# Configuration / external tools
# ---------------------------------------------------------------------------
GNINA_PATH = os.environ.get("GNINA_PATH", "/opt/gnina/gnina.1.3.2")
OBABEL_PATH = os.environ.get("OBABEL_PATH", "obabel")
LIGPREPPER_DIR = os.environ.get("LIGPREPPER_DIR", "/opt/webapps/ligprepper")

# Score fields where a *higher* value means a better pose.
_HIGHER_IS_BETTER_FIELDS = {"CNNscore", "CNNaffinity", "CNN_VS"}


class DockingCancelled(Exception):
    """Raised inside ``evaluate`` when a run has been asked to cancel."""


def score_field_is_higher_better(score_field: str) -> bool:
    """Return True when a larger value of ``score_field`` is a better pose."""
    return score_field in _HIGHER_IS_BETTER_FIELDS


# ---------------------------------------------------------------------------
# Ligand preparation (ported from gnina_webapp.prepare_single_ligand)
# ---------------------------------------------------------------------------
def _strip_sdf_properties(sdf_block: str) -> str:
    """Remove all SD properties from a mol block, keeping only the structure."""
    lines = sdf_block.split("\n")
    result_lines: List[str] = []
    in_properties = False
    for line in lines:
        if line.strip() == "M  END":
            result_lines.append(line)
            in_properties = True
            continue
        if in_properties:
            if line.strip().startswith(">"):
                continue
            if line.strip() == "$$$$":
                in_properties = False
                result_lines.append(line)
            continue
        result_lines.append(line)
    return "\n".join(result_lines)


def _rdkit_embed_and_minimize(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Embed a SMILES into 3D with ETKDGv3 and minimise with MMFF94s (UFF
    fallback). Returns ``(molblock_with_terminator, error)``.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"RDKit could not parse SMILES '{smiles}'"
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                return None, "ETKDG embedding failed"

        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props is not None else None
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(mol)
        if ff is not None:
            ff.Minimize(maxIts=2000)

        return Chem.MolToMolBlock(mol) + "$$$$\n", None
    except Exception as e:  # pragma: no cover - defensive
        return None, f"RDKit 3D generation error: {e}"


def prepare_ligand_3d(smiles: str, ph: float, identifier: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Protonate ``smiles`` at ``ph`` with OpenBabel, then build a single 3D
    conformer with RDKit (OpenBabel ``--gen3d`` fallback).

    Returns ``(sdf_block, error)``. ``sdf_block`` is a clean, property-free
    SDF record whose title line is ``identifier``.
    """
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="ts_lig_")
        smi_file = os.path.join(tmp_dir, "input.smi")
        protonated_smi_file = os.path.join(tmp_dir, "protonated.smi")

        with open(smi_file, "w") as f:
            f.write(f"{smiles} {identifier}\n")

        # Step 1: protonate at pH (-r strips salts / small fragments).
        cmd_protonate = [OBABEL_PATH, smi_file, "-O", protonated_smi_file, "-r", "-p", str(ph)]
        result1 = subprocess.run(cmd_protonate, capture_output=True, text=True, timeout=30)
        if not os.path.exists(protonated_smi_file) or os.path.getsize(protonated_smi_file) == 0:
            stderr_msg = (result1.stderr or "No output").strip()
            return None, f"Protonation failed for '{smiles}': {stderr_msg[:300]}"

        with open(protonated_smi_file) as f:
            protonated_line = f.readline().strip()
        protonated_smiles = re.split(r"[\s\t]", protonated_line, 1)[0] if protonated_line else ""
        if not protonated_smiles:
            return None, f"Protonation produced empty SMILES from '{smiles}'"

        # Step 2: 3D embed + minimise with RDKit (primary).
        sdf_block, rdkit_err = _rdkit_embed_and_minimize(protonated_smiles)

        # Fallback: OpenBabel --gen3d best.
        if sdf_block is None:
            sdf_file = os.path.join(tmp_dir, "output.sdf")
            cmd_3d = [OBABEL_PATH, protonated_smi_file, "-O", sdf_file, "--gen3d", "best"]
            result2 = subprocess.run(cmd_3d, capture_output=True, text=True, timeout=120)
            if os.path.exists(sdf_file) and os.path.getsize(sdf_file) > 50:
                with open(sdf_file) as f:
                    candidate = f.read()
                if candidate and "$$$$" in candidate and "M  END" in candidate:
                    sdf_block = candidate
            if sdf_block is None:
                ob_err = (result2.stderr or "unknown").strip()[:200]
                return None, f"3D generation failed for '{smiles}': RDKit={rdkit_err}; OpenBabel={ob_err}"

        lines = sdf_block.split("\n")
        if lines:
            lines[0] = identifier
        sdf_block = _strip_sdf_properties("\n".join(lines))
        return sdf_block, None

    except subprocess.TimeoutExpired:
        return None, f"Timeout preparing: {smiles[:50]}"
    except Exception as e:  # pragma: no cover - defensive
        return None, f"Error preparing {smiles[:50]}: {e}"
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Structural-alert + property filters (reuses ligprepper's SMARTS sets)
# ---------------------------------------------------------------------------
def _load_pains(path: Path) -> List[Tuple[str, Chem.Mol]]:
    """Load PAINS patterns from a tab-separated ``name<TAB>SMARTS`` file."""
    patterns: List[Tuple[str, Chem.Mol]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, smarts = parts[0].strip(), parts[1].strip()
            q = Chem.MolFromSmarts(smarts)
            if q is not None:
                patterns.append((name, q))
    return patterns


def _load_reos(path: Path) -> List[Tuple[int, str, Chem.Mol]]:
    """Load REOS rules from a ``SMARTS<TAB>max_count<TAB>description`` file."""
    rules: List[Tuple[int, str, Chem.Mol]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            smarts = parts[0].strip()
            desc = parts[2].strip().strip('"')
            try:
                max_count = int(parts[1].strip())
            except ValueError:
                continue
            q = Chem.MolFromSmarts(smarts)
            if q is not None:
                rules.append((max_count, desc, q))
    return rules


class MolFilters:
    """
    Pre-docking hard filters: PAINS / REOS structural alerts plus optional MW
    and logP ranges. ``reject_reason(mol)`` returns a short string describing
    the first failed filter, or ``None`` if the molecule passes everything.
    """

    def __init__(
        self,
        use_pains: bool = False,
        use_reos: bool = False,
        mw_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        logp_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ligprepper_dir: str = LIGPREPPER_DIR,
    ):
        self.mw_range = mw_range
        self.logp_range = logp_range
        self.pains_patterns: List[Tuple[str, Chem.Mol]] = []
        self.reos_rules: List[Tuple[int, str, Chem.Mol]] = []

        base = Path(ligprepper_dir)
        if use_pains:
            pf = base / "PAINS.txt"
            if pf.is_file():
                self.pains_patterns = _load_pains(pf)
            else:
                raise FileNotFoundError(f"PAINS file not found: {pf}")
        if use_reos:
            rf = base / "REOS.txt"
            if rf.is_file():
                self.reos_rules = _load_reos(rf)
            else:
                raise FileNotFoundError(f"REOS file not found: {rf}")

    @property
    def active(self) -> bool:
        return bool(
            self.pains_patterns or self.reos_rules or self.mw_range or self.logp_range
        )

    @staticmethod
    def _in_range(value: float, rng: Tuple[Optional[float], Optional[float]]) -> bool:
        lo, hi = rng
        if lo is not None and value < lo:
            return False
        if hi is not None and value > hi:
            return False
        return True

    def reject_reason(self, mol: Chem.Mol) -> Optional[str]:
        if self.mw_range is not None:
            mw = Descriptors.MolWt(mol)
            if not self._in_range(mw, self.mw_range):
                return f"MW {mw:.1f} outside {self.mw_range}"
        if self.logp_range is not None:
            logp = Crippen.MolLogP(mol)
            if not self._in_range(logp, self.logp_range):
                return f"logP {logp:.2f} outside {self.logp_range}"
        for name, q in self.pains_patterns:
            if mol.HasSubstructMatch(q):
                return f"PAINS: {name}"
        for max_count, desc, q in self.reos_rules:
            if len(mol.GetSubstructMatches(q)) > max_count:
                return f"REOS: {desc}"
        return None


# ---------------------------------------------------------------------------
# GNINA docking evaluator
# ---------------------------------------------------------------------------
class GninaEvaluator(Evaluator):
    """
    Score a molecule by docking it with GNINA and reading back a pose score.

    Required ``input_dict`` keys
        receptor_path : str   - prepared receptor PDB
    Binding site (exactly one of)
        reference_path : str  - reference ligand SDF for ``--autobox_ligand``
        center : (x, y, z) and optional size : (x, y, z) - explicit box
    Optional keys (defaults in parentheses)
        score_field ("minimizedAffinity"), cnn_scoring ("rescore"),
        exhaustiveness (8), num_modes (9), autobox_add (4.0), ph (7.4),
        box_size (16.0), gpu_id (0), work_dir (tempdir), gnina_path,
        filters (MolFilters instance), progress_callback (callable),
        timeout (600).
    """

    def __init__(self, input_dict: dict):
        self.receptor_path = input_dict["receptor_path"]
        if not os.path.isfile(self.receptor_path):
            raise FileNotFoundError(f"Receptor not found: {self.receptor_path}")

        self.reference_path = input_dict.get("reference_path")
        self.center = input_dict.get("center")
        size = input_dict.get("size")
        box_size = float(input_dict.get("box_size", 16.0))
        self.size = tuple(size) if size else (box_size, box_size, box_size)
        if not self.reference_path and self.center is None:
            raise ValueError("GninaEvaluator: provide reference_path or center")
        if self.reference_path and not os.path.isfile(self.reference_path):
            raise FileNotFoundError(f"Reference ligand not found: {self.reference_path}")

        self.score_field = input_dict.get("score_field", "minimizedAffinity")
        self.higher_is_better = score_field_is_higher_better(self.score_field)
        self.cnn_scoring = input_dict.get("cnn_scoring", "rescore")
        self.exhaustiveness = int(input_dict.get("exhaustiveness", 8))
        self.num_modes = int(input_dict.get("num_modes", 9))
        self.autobox_add = float(input_dict.get("autobox_add", 4.0))
        self.ph = float(input_dict.get("ph", 7.4))
        self.gpu_id = int(input_dict.get("gpu_id", os.environ.get("TS_DOCK_GPU", 0)))
        self.gnina_path = input_dict.get("gnina_path", GNINA_PATH)
        self.seed = int(input_dict.get("seed", 666))
        self.timeout = int(input_dict.get("timeout", 600))
        self.cpu = int(input_dict.get("cpu", 4))

        self.filters: Optional[MolFilters] = input_dict.get("filters")
        self.cancel_event = input_dict.get("cancel_event")  # threading.Event or None
        self.progress_callback: Optional[Callable[[str], None]] = input_dict.get("progress_callback")
        self.progress_every = int(input_dict.get("progress_every", 5))

        self.work_dir = input_dict.get("work_dir") or tempfile.mkdtemp(prefix="ts_gnina_")
        os.makedirs(self.work_dir, exist_ok=True)

        # State
        self.num_evaluations = 0
        self._dock_count = 0
        self._score_cache: Dict[str, float] = {}
        self._pose_cache: Dict[str, Tuple[float, Chem.Mol]] = {}  # smiles -> (score, pose mol)
        self.rejections: Dict[str, int] = {}
        self.prep_failures = 0
        self.dock_failures = 0
        self._best_score: Optional[float] = None

    # -- Evaluator interface ------------------------------------------------
    @property
    def counter(self) -> int:
        return self.num_evaluations

    def evaluate(self, mol: Chem.Mol) -> float:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise DockingCancelled()
        self.num_evaluations += 1
        try:
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            return np.nan

        cached = self._score_cache.get(smiles)
        if cached is not None:
            return cached

        # Hard filters before docking.
        if self.filters is not None and self.filters.active:
            reason = self.filters.reject_reason(mol)
            if reason is not None:
                key = reason.split(":")[0].split(" ")[0]
                self.rejections[key] = self.rejections.get(key, 0) + 1
                self._score_cache[smiles] = np.nan
                total_rej = sum(self.rejections.values())
                if self.progress_callback is not None and total_rej % 100 == 0:
                    self.progress_callback(
                        f"filtered {total_rej} products so far (pre-dock) {dict(self.rejections)}"
                    )
                return np.nan

        score = self._dock(smiles)
        self._score_cache[smiles] = score
        self._emit_progress(score)
        return score

    # -- Docking ------------------------------------------------------------
    def _dock(self, smiles: str) -> float:
        sdf_block, err = prepare_ligand_3d(smiles, self.ph, "ligand")
        if sdf_block is None:
            self.prep_failures += 1
            return np.nan

        self._dock_count += 1
        run_dir = os.path.join(self.work_dir, f"dock_{self._dock_count:06d}")
        os.makedirs(run_dir, exist_ok=True)
        lig_path = os.path.join(run_dir, "ligand.sdf")
        out_path = os.path.join(run_dir, "docked.sdf")
        with open(lig_path, "w") as f:
            f.write(sdf_block if sdf_block.endswith("\n") else sdf_block + "\n")

        cmd = [self.gnina_path, "-r", self.receptor_path, "-l", lig_path, "-o", out_path]
        if self.reference_path:
            cmd += ["--autobox_ligand", self.reference_path, "--autobox_add", str(self.autobox_add)]
        else:
            cx, cy, cz = self.center
            sx, sy, sz = self.size
            cmd += [
                "--center_x", f"{cx:.3f}", "--center_y", f"{cy:.3f}", "--center_z", f"{cz:.3f}",
                "--size_x", f"{sx:.3f}", "--size_y", f"{sy:.3f}", "--size_z", f"{sz:.3f}",
            ]
        cmd += [
            "--num_modes", str(self.num_modes),
            "--exhaustiveness", str(self.exhaustiveness),
            "--cnn_scoring", self.cnn_scoring,
            "--cpu", str(self.cpu),
            "--seed", str(self.seed),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            self.dock_failures += 1
            shutil.rmtree(run_dir, ignore_errors=True)
            return np.nan

        if proc.returncode != 0 or not os.path.isfile(out_path):
            self.dock_failures += 1
            shutil.rmtree(run_dir, ignore_errors=True)
            return np.nan

        score, pose_mol = self._best_pose(out_path, smiles)
        if pose_mol is not None:
            self._pose_cache[smiles] = (score, pose_mol)
        shutil.rmtree(run_dir, ignore_errors=True)
        return score if score is not None else np.nan

    def _best_pose(self, sdf_path: str, smiles: str) -> Tuple[Optional[float], Optional[Chem.Mol]]:
        """Read poses, return the best ``(score, mol)`` by the configured field."""
        best_score: Optional[float] = None
        best_mol: Optional[Chem.Mol] = None
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False)
        for pose in supplier:
            if pose is None:
                continue
            val = self._read_score(pose)
            if val is None or not np.isfinite(val):
                continue
            if best_score is None or (val > best_score if self.higher_is_better else val < best_score):
                best_score = val
                best_mol = pose
        if best_mol is None:
            return None, None
        best_mol.SetProp("_Name", smiles)
        best_mol.SetProp("SMILES", smiles)
        return best_score, best_mol

    def _read_score(self, pose: Chem.Mol) -> Optional[float]:
        # Try the configured field first, then minimizedAffinity as a fallback.
        for field in [self.score_field, "minimizedAffinity"]:
            if pose.HasProp(field):
                try:
                    return float(pose.GetProp(field))
                except (ValueError, TypeError):
                    continue
        return None

    # -- Progress + output --------------------------------------------------
    def _emit_progress(self, score: float) -> None:
        if np.isfinite(score):
            if self._best_score is None or (
                score > self._best_score if self.higher_is_better else score < self._best_score
            ):
                self._best_score = score
        if self.progress_callback is None:
            return
        if self._dock_count % self.progress_every == 0:
            best = f"{self._best_score:.3f}" if self._best_score is not None else "n/a"
            rej = sum(self.rejections.values())
            self.progress_callback(
                f"docked {self._dock_count} | best {self.score_field}={best} | "
                f"filtered {rej} | prep_fail {self.prep_failures} | dock_fail {self.dock_failures}"
            )

    def stats(self) -> dict:
        return {
            "evaluations": self.num_evaluations,
            "docked": self._dock_count,
            "unique_scored": len([s for s in self._score_cache.values() if np.isfinite(s)]),
            "rejections": dict(self.rejections),
            "prep_failures": self.prep_failures,
            "dock_failures": self.dock_failures,
            "best_score": self._best_score,
        }

    def write_top_poses(self, path: str, n: int = 100) -> int:
        """Write the best docked poses (sorted by score) to an SDF file."""
        poses = list(self._pose_cache.values())
        poses.sort(key=lambda x: x[0], reverse=self.higher_is_better)
        writer = Chem.SDWriter(path)
        written = 0
        try:
            for rank, (_score, mol) in enumerate(poses[:n], start=1):
                mol.SetProp("DockingRank", str(rank))
                writer.write(mol)
                written += 1
        finally:
            writer.close()
        return written
