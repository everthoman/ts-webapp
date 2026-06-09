#!/usr/bin/env python
"""Build annotated reagent tables from a raw pool + ``reactions.json``.

This is the *authoring* layer that compiles down to the per-component ``.smi``
lists the Thompson Sampling sampler reads at run time. It classifies every
reagent by:

* the reactive **handles** it carries (curated SMARTS — ``primary_amine``,
  ``carboxylic_acid``, ``aryl_halide``, ``boronic`` …), and
* which reaction **components** it can serve, *derived* by matching the
  reaction's own reactant template (so the role columns can never drift out of
  sync with ``reactions.json``).

It then flags **difunctional** building blocks (carry >=2 orthogonal handle
families — the ones that can sit *inside* a multi-step route, e.g. your
amino-benzoic acids) and homo-difunctional **conflicts** (e.g. a di-amine,
where it is ambiguous which end reacts), and emits:

  1. a master CSV (one row per reagent) — the single source of truth; and
  2. per-handle ``.smi`` *views*, and with ``--catalog-sets`` the exact
     ``data/*.smi`` files ``reactions.json`` references — generated and
     validated rather than hand-maintained.

Multi-step "suitable for a 3-step order" is deliberately **not** baked in as a
static column: it depends on the chosen reactions, their order and any
protecting-group strategy, and a reagent's role flips with position (a
Boc-amino-acid is a step-1 acid, then a step-2 amine once deprotected). We store
the atomic facts (handles + difunctional class) and let the route logic /
``/extend-options`` chain-rate preflight combine them per actual route.

Usage
-----
    conda activate ts_gnina
    # annotate a raw pool -> master CSV + per-handle view files
    python build_reagents.py pool.smi -o data --master data/reagents_master.csv
    # also regenerate the catalog's data/*.smi from the pool
    python build_reagents.py pool*.smi -o data --catalog-sets
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Reactive-handle definitions (curated). These mirror the reactant patterns in
# reactions.json but are deduped to chemistry-level groups, plus a couple the
# reaction SMARTS are deliberately permissive about and must be *curated*:
# `activated_aryl_halide` is the classic case — SNAr's template matches any
# aryl-Cl/F, but only EWG-activated ones actually react.
# ---------------------------------------------------------------------------
HANDLE_SMARTS: Dict[str, str] = {
    "primary_amine":   "[NX3;H2;!$([NX3]C=O);!$([NX3]=*);!$([NX3]S(=O)=O)][#6]",
    "secondary_amine": "[NX3;H1;!$([NX3]C=O);!$([NX3]=*);!$([NX3]S(=O)=O)]([#6])[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "aryl_halide":     "[c][F,Cl,Br,I]",
    # Heuristic: an aryl-F/Cl that is EWG-activated (ortho/para to NO2, C#N or
    # C=O) or sits on an azine ring — i.e. actually competent for SNAr. Tune me.
    "activated_aryl_halide":
        "[$([F,Cl][c]:[n]),"
        "$([F,Cl][c]:c:[n]),"
        "$([F,Cl][c]:c:c:[n]),"
        "$([F,Cl][c]1:c:c:c:c:c:1[$([NX3+](=O)[O-]),$(C#N),$([CX3]=O)]),"
        "$([F,Cl][c]1:c:c:c:c:c:1[c][$([NX3+](=O)[O-]),$(C#N)])]",
    "boronic":         "[#6][BX3]([OX2])[OX2]",
    "aldehyde":        "[CX3H1](=O)[#6]",
    "ketone":          "[#6][CX3](=O)[#6]",
}

# Handle -> family. Difunctionality / conflicts are judged per family so that an
# amino-acid (amine + acid = two families) reads as a chainable linker, while a
# di-amine (two of one family) reads as an ambiguous conflict.
HANDLE_FAMILY: Dict[str, str] = {
    "primary_amine": "amine",
    "secondary_amine": "amine",
    "carboxylic_acid": "acid",
    "aryl_halide": "aryl_halide",
    "boronic": "boronic",
    "aldehyde": "carbonyl",
    "ketone": "carbonyl",
}

# Refinement handles are *subsets* of a counted handle (an activated aryl halide
# is also an aryl halide), so they're recorded as tags but excluded from the
# family tally — otherwise one halide would double-count as a false conflict.
REFINEMENT_HANDLES = {"activated_aryl_halide"}

# ---------------------------------------------------------------------------
# Catalog set specs: how to regenerate each data/*.smi from the annotated pool.
# `require` / `exclude` are handle names; `aromatic_acid` requires the acid to
# sit on an aromatic ring (benzoic). `limit` truncates (the _100/_500 variants
# are just the head of the full list, matching the current catalog). This is the
# bit that turns the catalog from hand-maintained into generated.
# ---------------------------------------------------------------------------
AROMATIC_ACID = Chem.MolFromSmarts("[c][CX3](=O)[OX2H1]")

CATALOG_SETS: Dict[str, dict] = {
    "primary_amines_ok":          {"require": ["primary_amine"], "exclude": ["carboxylic_acid"], "no_conflict": True},
    "primary_amines_500":         {"like": "primary_amines_ok", "limit": 500},
    "primary_amines_100":         {"like": "primary_amines_ok", "limit": 100},
    "carboxylic_acids_ok":        {"require": ["carboxylic_acid"], "exclude": ["primary_amine", "secondary_amine"], "no_conflict": True},
    "carboxylic_acids_500":       {"like": "carboxylic_acids_ok", "limit": 500},
    "carboxylic_acids_100":       {"like": "carboxylic_acids_ok", "limit": 100},
    "aminobenzoic_ok":            {"require": ["primary_amine", "carboxylic_acid"], "aromatic_acid": True},
    "aminobenzoic_100":           {"like": "aminobenzoic_ok", "limit": 100},
    "aryl_halides_100":           {"require": ["aryl_halide"], "limit": 100},
    "boronic_acids_100":          {"require": ["boronic"], "limit": 100},
    "activated_aryl_halides_100": {"require": ["activated_aryl_halide"], "limit": 100},
    "aldehydes_ketones_100":      {"require": ["aldehyde", "ketone"], "require_any": True, "limit": 100},
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def compile_handles() -> Dict[str, Chem.Mol]:
    out = {}
    for name, sm in HANDLE_SMARTS.items():
        q = Chem.MolFromSmarts(sm)
        if q is None:
            raise ValueError(f"Bad handle SMARTS for {name!r}: {sm}")
        out[name] = q
    return out


def reaction_component_queries(reactions: List[dict]) -> List[Tuple[str, str, Chem.Mol]]:
    """Return ``(reaction_id, column_name, reactant_template)`` for every
    reaction *component* (the new-reagent inputs), derived from the reaction's
    own SMARTS so role columns can't drift from reactions.json.

    Convention (holds for all current reactions): an extend reaction's reactant
    templates are ``[intermediate, component_0, component_1, …]`` — the running
    intermediate comes first — so the components map to the *trailing* templates.
    """
    cols: List[Tuple[str, str, Chem.Mol]] = []
    for r in reactions:
        rxn = AllChem.ReactionFromSmarts(r["smarts"])
        n = rxn.GetNumReactantTemplates()
        comps = r["components"]
        offset = n - len(comps)
        if offset < 0:
            print(f"WARN: reaction {r['id']} has {n} reactant templates "
                  f"but {len(comps)} components; skipping", file=sys.stderr)
            continue
        for i, c in enumerate(comps):
            # Re-parse the template into a standalone query: the Mol returned by
            # GetReactantTemplate is owned by `rxn` and dangles (segfaults) once
            # the reaction is garbage-collected, so we detach it via SMARTS.
            tmpl = Chem.MolFromSmarts(Chem.MolToSmarts(rxn.GetReactantTemplate(offset + i)))
            cols.append((r["id"], f"{r['id']}.{_slug(c['label'])}", tmpl))
    return cols


def read_pool(paths: List[Path]) -> List[Tuple[str, str]]:
    """Read ``SMILES name`` lines from the input file(s), dedupe by canonical
    SMILES, keep the first name seen."""
    seen: Dict[str, str] = {}
    n_lines = n_bad = 0
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                n_lines += 1
                parts = line.split()
                smi = parts[0]
                name = parts[1] if len(parts) > 1 else f"reagent_{n_lines}"
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    n_bad += 1
                    continue
                can = Chem.MolToSmiles(mol)
                seen.setdefault(can, name)
    print(f"Read {n_lines} lines from {len(paths)} file(s): "
          f"{len(seen)} unique parseable, {n_bad} unparseable.", file=sys.stderr)
    return list(seen.items())


def classify(smiles: str, handles: Dict[str, Chem.Mol],
             role_cols: List[Tuple[str, str, Chem.Mol]]) -> Optional[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Handle counts (number of matches, so we can spot homo-difunctional cases).
    counts = {h: len(mol.GetSubstructMatches(q)) for h, q in handles.items()}
    present = [h for h, c in counts.items() if c]
    fam_counts: Counter = Counter()
    for h in present:
        if h in REFINEMENT_HANDLES:
            continue
        fam_counts[HANDLE_FAMILY[h]] += counts[h]
    families = sorted(fam_counts)
    difunctional = len(families) >= 2
    conflict = any(v > 1 for v in fam_counts.values())
    roles = {col: int(mol.HasSubstructMatch(tmpl)) for _rid, col, tmpl in role_cols}
    return {
        "SMILES": smiles,
        "MW": round(Descriptors.MolWt(mol), 1),
        "logP": round(Crippen.MolLogP(mol), 2),
        "handles": ";".join(present),
        "families": ";".join(families),
        "n_families": len(families),
        "difunctional": int(difunctional),
        "conflict": int(conflict),
        "aromatic_acid": int(mol.HasSubstructMatch(AROMATIC_ACID)),
        **{h: counts[h] for h in handles},
        **roles,
    }


def write_master(rows: List[dict], handles, role_cols, path: Path) -> None:
    role_names = [c for _rid, c, _q in role_cols]
    fields = (["name", "SMILES", "MW", "logP", "handles", "families", "n_families",
               "difunctional", "conflict", "aromatic_acid"]
              + list(handles) + role_names)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote master table: {path}  ({len(rows)} reagents, {len(fields)} columns)")


def select_set(rows: List[dict], spec: dict, resolved: Dict[str, List[dict]]) -> List[dict]:
    if "like" in spec:
        base = resolved[spec["like"]]
        out = base
    else:
        require = spec.get("require", [])
        exclude = spec.get("exclude", [])
        any_mode = spec.get("require_any", False)
        out = []
        for r in rows:
            has_req = [r.get(h, 0) > 0 for h in require]
            ok = (any(has_req) if any_mode else all(has_req)) if require else True
            if ok and any(r.get(h, 0) > 0 for h in exclude):
                ok = False
            if ok and spec.get("no_conflict") and r["conflict"]:
                ok = False
            if ok and spec.get("aromatic_acid") and not r["aromatic_acid"]:
                ok = False
            if ok:
                out.append(r)
    if spec.get("limit"):
        out = out[: spec["limit"]]
    return out


def write_smi(rows: List[dict], path: Path) -> None:
    with open(path, "w") as fh:
        for r in rows:
            fh.write(f"{r['SMILES']} {r['name']}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="+", help="raw reagent .smi file(s) ('SMILES name' per line)")
    ap.add_argument("-r", "--reactions", default=str(BASE_DIR / "reactions.json"))
    ap.add_argument("-o", "--out-dir", default=str(BASE_DIR / "data"))
    ap.add_argument("--master", default=None, help="master CSV path (default: <out-dir>/reagents_master.csv)")
    ap.add_argument("--views", action="store_true", help="also emit per-handle gen_<handle>.smi views")
    ap.add_argument("--catalog-sets", action="store_true",
                    help="regenerate the data/*.smi files reactions.json references")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = Path(args.master) if args.master else out_dir / "reagents_master.csv"

    catalog = json.loads(Path(args.reactions).read_text())
    handles = compile_handles()
    role_cols = reaction_component_queries(catalog["reactions"])

    pool = read_pool([Path(p) for p in args.pool])
    rows: List[dict] = []
    for smi, name in pool:
        rec = classify(smi, handles, role_cols)
        if rec is not None:
            rec["name"] = name
            rows.append(rec)

    write_master(rows, handles, role_cols, master_path)

    # Summary by handle.
    print("\nHandle inventory:")
    for h in handles:
        print(f"  {h:24s} {sum(1 for r in rows if r[h] > 0):6d}")
    print(f"  {'difunctional':24s} {sum(r['difunctional'] for r in rows):6d}")
    print(f"  {'conflict (homo-difunc)':24s} {sum(r['conflict'] for r in rows):6d}")

    if args.views:
        for h in handles:
            sel = [r for r in rows if r[h] > 0]
            write_smi(sel, out_dir / f"gen_{h}.smi")
        write_smi([r for r in rows if r["difunctional"]], out_dir / "gen_difunctional.smi")
        print("Wrote per-handle view files (gen_*.smi).")

    if args.catalog_sets:
        print("\nRegenerating catalog sets:")
        resolved: Dict[str, List[dict]] = {}
        for set_id, spec in CATALOG_SETS.items():
            sel = select_set(rows, spec, resolved)
            resolved[set_id] = sel
            fname = catalog["reagent_sets"].get(set_id, {}).get("file", f"data/{set_id}.smi")
            path = BASE_DIR / fname
            write_smi(sel, path)
            print(f"  {set_id:30s} {len(sel):6d} -> {fname}")


if __name__ == "__main__":
    main()
