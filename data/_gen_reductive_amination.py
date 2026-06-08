"""Generate a small example carbonyl library for the reductive amination reaction.

Produces (in this directory):
  aldehydes_ketones_100.smi   aldehydes + ketones (reductive amination electrophile)

The amine partner reuses the existing primary_amines_* sets. Every SMILES is
round-tripped through RDKit, de-duplicated on canonical SMILES, and the
reductive amination SMARTS is test-fired so we ship only reactive carbonyls.
"""
from itertools import product
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

HERE = Path(__file__).resolve().parent

# Carbonyl = aldehyde or ketone only (recursive SMARTS excludes acids/esters/
# amides); amine = primary/secondary, not an amide N. Carbonyl C -> CH bonded to N.
REDUCTIVE_AMINATION = (
    "[$([CX3H1]=O),$([CX3]([#6])([#6])=O);"
    "!$([CX3](=O)[OX2]);!$([CX3](=O)[#7]):1]=[OX1]."
    "[NX3;H1,H2;!$([NX3]C=O):2]>>[C:1][N:2]"
)

SUBSTS = ["", "C", "CC", "C(C)C", "OC", "OCC", "F", "Cl", "C#N", "C(F)(F)F",
          "C(C)=O", "N(C)C", "NC(C)=O", "S(C)(=O)=O", "C(=O)OC",
          "C(N)=O", "CO", "c1ccccc1", "c1ccncc1"]


def _decorate(cores):
    out = []
    for core, s in product(cores, SUBSTS):
        smi = core.replace("{s}", s) if "{s}" in core else core
        smi = smi.replace("()", "")
        out.append(smi)
    return out


def canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(mol)


def carbonyls():
    cores = [
        # aromatic / heteroaromatic aldehydes
        "O=Cc1ccccc1{s}",
        "O=Cc1ccc({s})cc1",
        "O=Cc1cccc({s})c1",
        "O=Cc1ccncc1",
        "O=Cc1cccnc1",
        "O=Cc1ccc(nc1){s}",
        "O=Cc1ccc(s1){s}",
        "O=Cc1ccc(o1){s}",
        "O=Cc1ccc2ccccc2c1",
        # aliphatic aldehydes
        "O=CC{s}",
        "O=CCC{s}",
        "O=CCCC{s}",
        "O=CC1CCCCC1",
        "O=CCc1ccccc1{s}",
        # ketones
        "O=C(C)c1ccccc1{s}",
        "O=C(C)C{s}",
        "O=C1CCCCC1",
        "O=C1CCNCC1",
        "O=C(c1ccccc1){s}",
    ]
    return _decorate(cores)


def dedup_canon(smiles, limit=100):
    seen = {}
    for smi in smiles:
        c = canon(smi)
        if c and c not in seen:
            seen[c] = True
        if len(seen) >= limit:
            break
    return list(seen)


def fires(rxn, reactant_smis, partner_smi):
    partner = Chem.MolFromSmiles(partner_smi)
    good = []
    for smi in reactant_smis:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for order in ((m, partner), (partner, m)):
            ok = False
            for p in rxn.RunReactants(order):
                try:
                    Chem.SanitizeMol(p[0])
                    ok = True
                    break
                except Exception:
                    continue
            if ok:
                good.append(smi)
                break
    return good


def write_smi(path, smiles, prefix):
    lines = [f"{smi} {prefix}_{i:03d}" for i, smi in enumerate(smiles, 1)]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(smiles):3d}  ->  {path.name}")


def main():
    rxn = AllChem.ReactionFromSmarts(REDUCTIVE_AMINATION)
    pool = dedup_canon(carbonyls(), 300)
    good = fires(rxn, pool, "NCc1ccccc1")[:100]
    write_smi(HERE / "aldehydes_ketones_100.smi", good, "carb")

    p = rxn.RunReactants((Chem.MolFromSmiles(good[0]),
                          Chem.MolFromSmiles("NCc1ccccc1")))[0][0]
    Chem.SanitizeMol(p)
    print("Reductive amination sample product:", Chem.MolToSmiles(p))


if __name__ == "__main__":
    main()
