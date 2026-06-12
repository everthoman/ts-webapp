#!/usr/bin/env python
"""Tag a reagent pool against a functional-group vocabulary + ``reactions.json``.

This is the *authoring* layer for the synthon-style model: a single reagent pool
is tagged once against a shared functional-group vocabulary, and reactions
reference those class names rather than curated per-reaction files — so adding a
reaction (or swapping a 15K in-house pool for a 130K one) needs no reagent
re-curation. See ``functional_groups.json`` for the vocabulary.

For every reagent it records:

* the **functional-group tags** it carries (SMARTS match against the vocabulary,
  e.g. ``primary_amine``, ``carboxylic_acid``, ``activated_aryl_halide``), and
* which reaction **components** it can serve, *derived* by matching the
  reaction's own reactant template (so role columns can't drift from
  ``reactions.json``).

It flags **difunctional** building blocks (>=2 orthogonal handle families — the
ones that can sit *inside* a multi-step route) and homo-difunctional
**conflicts** (e.g. a di-amine), and emits:

  1. a **tagged registry** (one row per block: stable id, SMILES, name, source,
     price placeholder, fg_tags, …) — the single source of truth; and
  2. an **inverted index** (class -> block ids) for O(1) per-component pruning at
     route time.

Optionally (``--master``) a detailed CSV with per-class counts and reaction-role
columns, and (``--catalog-sets``) regenerated ``data/*.smi`` files.

Multi-step "suitable for a 3-step order" is deliberately **not** baked in as a
static column: it depends on the chosen reactions, their order and any
protecting-group strategy. We store the atomic facts (tags + difunctional class)
and let the route logic / ``/extend-options`` chain-rate preflight combine them
per actual route.

Usage
-----
    conda activate ts_gnina
    # tag a pool -> registry + inverted index
    python build_reagents.py inhouse.smi --source inhouse -o data
    # also emit the detailed master table and regenerate catalog .smi
    python build_reagents.py pool*.smi --master data/reagents_master.csv --catalog-sets
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VOCAB = BASE_DIR / "functional_groups.json"

# Used by the catalog-set specs to tell an aromatic (benzoic) acid from an
# aliphatic one — a curation detail, not a vocabulary class.
AROMATIC_ACID = Chem.MolFromSmarts("[c][CX3](=O)[OX2H1]")

# ---------------------------------------------------------------------------
# Catalog set specs: how to regenerate each data/*.smi from the tagged pool.
# `require` / `exclude` are class names; `aromatic_acid` requires the acid on an
# aromatic ring; `limit` truncates (the _100/_500 variants are just the head).
# This turns the catalog from hand-maintained into generated.
# ---------------------------------------------------------------------------
CATALOG_SETS: Dict[str, dict] = {
    "primary_amines_ok":          {"require": ["primary_amine"], "exclude": ["carboxylic_acid"], "no_conflict": True},
    "primary_amines_500":         {"like": "primary_amines_ok", "limit": 500},
    "primary_amines_100":         {"like": "primary_amines_ok", "limit": 100},
    "secondary_amines_ok":        {"require": ["secondary_amine"], "exclude": ["carboxylic_acid"], "no_conflict": True},
    "secondary_amines_500":       {"like": "secondary_amines_ok", "limit": 500},
    "secondary_amines_100":       {"like": "secondary_amines_ok", "limit": 100},
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


# ---------------------------------------------------------------------------
# Functional-group vocabulary
# ---------------------------------------------------------------------------
class Vocab:
    """The functional-group vocabulary loaded from functional_groups.json.

    Holds, per class: a compiled SMARTS query, its family (for the
    difunctional/conflict tally) and whether it is a *refinement* (a subset of a
    parent class, e.g. activated aryl halide ⊂ aryl halide) — refinements are
    tags but are excluded from the family tally so they don't double-count.
    """

    def __init__(self, path: Path):
        doc = json.loads(Path(path).read_text())
        self.version = doc.get("version")
        self.groups = doc["groups"]
        self.query: Dict[str, Chem.Mol] = {}
        self.family: Dict[str, str] = {}
        self.refinement: set = set()
        for name, g in self.groups.items():
            q = Chem.MolFromSmarts(g["smarts"])
            if q is None:
                raise ValueError(f"Bad SMARTS for functional group {name!r}: {g['smarts']}")
            self.query[name] = q
            self.family[name] = g.get("family", name)
            if g.get("refinement"):
                self.refinement.add(name)

    @property
    def names(self) -> List[str]:
        return list(self.query)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def bb_id(canonical_smiles: str) -> str:
    """Stable, content-addressed building-block id (so pools merge/dedupe cleanly
    across runs and sources)."""
    return "BB" + hashlib.sha1(canonical_smiles.encode()).hexdigest()[:10]


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


def _enamine_id_key(name: str):
    """Sort key for Enamine catalog IDs: numeric value if purely numeric, else the
    raw string, so the lowest catalog number wins when deduplicating."""
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def read_pool(paths: List[Path]) -> List[Tuple[str, str]]:
    """Read ``SMILES name`` lines from the input file(s).

    Each molecule is salt-stripped (largest fragment kept), neutralized, then
    canonicalized.  When multiple input lines map to the same canonical SMILES
    the one with the lowest Enamine catalog ID (numeric sort) is kept.
    """
    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()

    # canonical SMILES -> (name, id_key)
    seen: Dict[str, Tuple[str, tuple]] = {}
    n_lines = n_bad = n_salt = n_neutralized = 0
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
                # Salt stripping: keep largest fragment
                stripped = chooser.choose(mol)
                if stripped.GetNumAtoms() < mol.GetNumAtoms():
                    n_salt += 1
                mol = stripped
                # Neutralize formal charges where possible
                uncharged = uncharger.uncharge(mol)
                if Chem.MolToSmiles(uncharged) != Chem.MolToSmiles(mol):
                    n_neutralized += 1
                mol = uncharged
                can = Chem.MolToSmiles(mol)
                key = _enamine_id_key(name)
                if can not in seen or key < seen[can][1]:
                    seen[can] = (name, key)
    print(f"Read {n_lines} lines from {len(paths)} file(s): "
          f"{n_salt} salt-stripped, {n_neutralized} neutralized, "
          f"{len(seen)} unique after dedup, {n_bad} unparseable.", file=sys.stderr)
    return [(can, name) for can, (name, _key) in seen.items()]


def classify(smiles: str, vocab: Vocab,
             role_cols: List[Tuple[str, str, Chem.Mol]]) -> Optional[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Per-class match counts (so we can spot homo-difunctional cases).
    counts = {name: len(mol.GetSubstructMatches(q)) for name, q in vocab.query.items()}
    present = [name for name, c in counts.items() if c]
    fam_counts: Counter = Counter()
    for name in present:
        if name in vocab.refinement:
            continue  # refinements are subsets of their parent; don't double-count
        fam_counts[vocab.family[name]] += counts[name]
    families = sorted(fam_counts)
    difunctional = len(families) >= 2
    conflict = any(v > 1 for v in fam_counts.values())
    roles = {col: int(mol.HasSubstructMatch(tmpl)) for _rid, col, tmpl in role_cols}
    return {
        "id": bb_id(smiles),
        "SMILES": smiles,
        "MW": round(Descriptors.MolWt(mol), 1),
        "logP": round(Crippen.MolLogP(mol), 2),
        "fg_tags": ";".join(present),
        "families": ";".join(families),
        "n_families": len(families),
        "difunctional": int(difunctional),
        "conflict": int(conflict),
        "aromatic_acid": int(mol.HasSubstructMatch(AROMATIC_ACID)),
        **{name: counts[name] for name in vocab.names},
        **roles,
    }


def write_registry(rows: List[dict], path: Path) -> None:
    """The tagged registry — the per-block source of truth. `price` is a
    placeholder column now so cost can be folded in later without a schema
    change (product cost = sum of block prices + optional per-step cost)."""
    fields = ["id", "SMILES", "name", "source", "price", "MW", "logP",
              "fg_tags", "families", "n_families", "difunctional", "conflict"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote registry: {path}  ({len(rows)} blocks)")


def build_index(rows: List[dict], vocab: Vocab) -> dict:
    """Inverted index class -> [block ids] for O(1) per-component pruning, plus a
    difunctional id list for internal-junction reasoning."""
    by_class: Dict[str, List[str]] = {name: [] for name in vocab.names}
    difunctional: List[str] = []
    for r in rows:
        for name in vocab.names:
            if r[name] > 0:
                by_class[name].append(r["id"])
        if r["difunctional"]:
            difunctional.append(r["id"])
    return {
        "version": 1,
        "vocab_version": vocab.version,
        "n_reagents": len(rows),
        "counts": {name: len(ids) for name, ids in by_class.items()},
        "by_class": by_class,
        "difunctional": difunctional,
    }


def write_master(rows: List[dict], vocab: Vocab, role_cols, path: Path) -> None:
    role_names = [c for _rid, c, _q in role_cols]
    fields = (["id", "name", "source", "price", "SMILES", "MW", "logP", "fg_tags",
               "families", "n_families", "difunctional", "conflict", "aromatic_acid"]
              + vocab.names + role_names)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote master table: {path}  ({len(rows)} reagents, {len(fields)} columns)")


def select_set(rows: List[dict], spec: dict, resolved: Dict[str, List[dict]]) -> List[dict]:
    if "like" in spec:
        out = resolved[spec["like"]]
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
    ap.add_argument("-g", "--vocab", default=str(DEFAULT_VOCAB), help="functional-group vocabulary JSON")
    ap.add_argument("-o", "--out-dir", default=str(BASE_DIR / "data"))
    ap.add_argument("--source", default="inhouse", help="provenance tag for these blocks (e.g. inhouse, enamine)")
    ap.add_argument("--registry", default=None, help="registry CSV (default: <out-dir>/reagents_registry.csv)")
    ap.add_argument("--index", default=None, help="inverted-index JSON (default: <out-dir>/reagents_index.json)")
    ap.add_argument("--master", default=None, help="also write a detailed master CSV (per-class counts + role columns)")
    ap.add_argument("--views", action="store_true", help="also emit per-class gen_<class>.smi views")
    ap.add_argument("--catalog-sets", action="store_true",
                    help="regenerate the data/*.smi files reactions.json references")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.registry) if args.registry else out_dir / "reagents_registry.csv"
    index_path = Path(args.index) if args.index else out_dir / "reagents_index.json"

    catalog = json.loads(Path(args.reactions).read_text())
    vocab = Vocab(Path(args.vocab))
    role_cols = reaction_component_queries(catalog["reactions"])
    print(f"Vocabulary: {len(vocab.names)} classes (v{vocab.version}) from {args.vocab}", file=sys.stderr)

    pool = read_pool([Path(p) for p in args.pool])
    rows: List[dict] = []
    for smi, name in pool:
        rec = classify(smi, vocab, role_cols)
        if rec is not None:
            rec["name"] = name
            rec["source"] = args.source
            rec["price"] = ""  # filled in later; product cost = sum of block prices
            rows.append(rec)

    write_registry(rows, registry_path)
    index = build_index(rows, vocab)
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Wrote inverted index: {index_path}")

    # Summary by class.
    print("\nFunctional-group inventory:")
    for name in vocab.names:
        print(f"  {name:24s} {index['counts'][name]:6d}")
    print(f"  {'difunctional':24s} {sum(r['difunctional'] for r in rows):6d}")
    print(f"  {'conflict (homo-difunc)':24s} {sum(r['conflict'] for r in rows):6d}")

    if args.master:
        write_master(rows, vocab, role_cols, Path(args.master))

    if args.views:
        for name in vocab.names:
            write_smi([r for r in rows if r[name] > 0], out_dir / f"gen_{name}.smi")
        write_smi([r for r in rows if r["difunctional"]], out_dir / "gen_difunctional.smi")
        print("Wrote per-class view files (gen_*.smi).")

    if args.catalog_sets:
        print("\nRegenerating catalog sets:")
        resolved: Dict[str, List[dict]] = {}
        for set_id, spec in CATALOG_SETS.items():
            sel = select_set(rows, spec, resolved)
            resolved[set_id] = sel
            fname = catalog["reagent_sets"].get(set_id, {}).get("file", f"data/{set_id}.smi")
            write_smi(sel, BASE_DIR / fname)
            print(f"  {set_id:30s} {len(sel):6d} -> {fname}")


if __name__ == "__main__":
    main()
