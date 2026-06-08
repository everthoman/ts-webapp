"""Generate small example reagent libraries for the Suzuki and SNAr reactions.

Produces (in this directory):
  aryl_halides_100.smi        aryl/heteroaryl bromides (Suzuki electrophile)
  boronic_acids_100.smi       aryl/heteroaryl boronic acids (Suzuki nucleophile)
  activated_aryl_halides_100.smi  EWG-activated aryl fluorides/chlorides (SNAr electrophile)

Every SMILES is round-tripped through RDKit, de-duplicated on canonical SMILES,
and the Suzuki / SNAr reaction SMARTS are test-fired against the libraries so we
ship only reagents that actually react.
"""
from itertools import product
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

HERE = Path(__file__).resolve().parent

# Reaction SMARTS that ts_webapp.py / reactions.json will use.
SUZUKI = "[c:1][Cl,Br,I].[c:2][B]([OX2])[OX2]>>[c:1][c:2]"
SNAR = "[c:1][F,Cl].[NX3;H1,H2:2]>>[c:1][N:2]"

# Substituent decorations applied to an aromatic ring (as SMILES fragments).
SUBSTS = ["", "C", "CC", "C(C)C", "OC", "OCC", "F", "Cl", "C#N", "C(F)(F)F",
          "C(C)=O", "C(=O)C", "N(C)C", "NC(C)=O", "S(C)(=O)=O", "C(=O)OC",
          "C(N)=O", "CO", "c1ccccc1", "c1ccncc1"]


def _decorate(cores):
    """Apply each substituent to every {s} slot, dropping empty-group artifacts."""
    out = []
    for core, s in product(cores, SUBSTS):
        smi = core.replace("{s}", s) if "{s}" in core else core
        smi = smi.replace("()", "")  # ({s}) with s="" -> drop the empty parens
        out.append(smi)
    return out


def canon(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def aryl_halides():
    """Aryl / heteroaryl bromides for Suzuki coupling."""
    # cores carry one Br (the coupling site); {s} marks a decoration position.
    cores = [
        "Brc1ccccc1{s}",
        "Brc1ccc({s})cc1",
        "Brc1cccc({s})c1",
        "Brc1ccc(cc1){s}",
        "Brc1ccncc1",          # 4-bromopyridine
        "Brc1cccnc1",          # 3-bromopyridine
        "Brc1ccc(nc1){s}",
        "Brc1ccc2ccccc2c1",    # bromonaphthalene
        "Brc1ccc(cc1)c1ccccc1",
        "Brc1cc2ccccc2[nH]1",  # bromoindole
        "Brc1ccc(s1){s}",      # bromothiophene
        "Brc1ccc(o1){s}",      # bromofuran
    ]
    return _decorate(cores)


def boronic_acids():
    """Aryl / heteroaryl boronic acids for Suzuki coupling."""
    cores = [
        "OB(O)c1ccccc1{s}",
        "OB(O)c1ccc({s})cc1",
        "OB(O)c1cccc({s})c1",
        "OB(O)c1ccncc1",
        "OB(O)c1cccnc1",
        "OB(O)c1ccc(nc1){s}",
        "OB(O)c1ccc2ccccc2c1",
        "OB(O)c1ccc(s1){s}",
        "OB(O)c1ccc(o1){s}",
        "OB(O)c1cc2ccccc2[nH]1",
    ]
    return _decorate(cores)


def activated_aryl_halides():
    """Aryl fluorides/chlorides activated toward SNAr (ortho/para EWG, azines)."""
    cores = [
        # nitro-activated halobenzenes
        "Fc1ccc([N+](=O)[O-])cc1{s}",
        "Fc1ccc([N+](=O)[O-])cc1",
        "Fc1cccc([N+](=O)[O-])c1",
        "Clc1ccc([N+](=O)[O-])cc1{s}",
        "Clc1cccc([N+](=O)[O-])c1",
        # cyano / carbonyl / sulfonyl activated
        "Fc1ccc(C#N)cc1{s}",
        "Fc1ccc(C(F)(F)F)cc1",
        "Fc1ccc(C(C)=O)cc1",
        "Fc1ccc(S(C)(=O)=O)cc1",
        "Fc1ccc(C(=O)OC)cc1",
        # electron-poor azines (intrinsically SNAr-active)
        "Fc1ccncc1{s}",
        "Fc1ccncc1",
        "Clc1ccncc1",
        "Fc1ncccn1",            # 2-fluoropyrimidine
        "Clc1ncccn1",
        "Fc1ccc(nc1){s}",
        "Clc1nccnc1",           # chloropyrazine
        "Fc1ccnc(n1){s}",
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
    """Keep only reactant_smis that yield a sanitizable product with partner."""
    partner = Chem.MolFromSmiles(partner_smi)
    good = []
    for smi in reactant_smis:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        for order in ((m, partner), (partner, m)):
            prods = rxn.RunReactants(order)
            ok = False
            for p in prods:
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
    suzuki = AllChem.ReactionFromSmarts(SUZUKI)
    snar = AllChem.ReactionFromSmarts(SNAR)

    halides = dedup_canon(aryl_halides(), 200)
    boronics = dedup_canon(boronic_acids(), 200)
    activated = dedup_canon(activated_aryl_halides(), 200)

    # Test-fire against a representative partner, keep only reactive members.
    halides = fires(suzuki, halides, "OB(O)c1ccccc1")[:100]
    boronics = fires(suzuki, boronics, "Brc1ccccc1")[:100]
    activated = fires(snar, activated, "NCc1ccccc1")[:100]

    write_smi(HERE / "aryl_halides_100.smi", halides, "arX")
    write_smi(HERE / "boronic_acids_100.smi", boronics, "boro")
    write_smi(HERE / "activated_aryl_halides_100.smi", activated, "actX")

    # Sanity: show one product from each reaction.
    p = suzuki.RunReactants((Chem.MolFromSmiles(halides[0]),
                             Chem.MolFromSmiles(boronics[0])))[0][0]
    Chem.SanitizeMol(p)
    print("Suzuki sample product:", Chem.MolToSmiles(p))
    p = snar.RunReactants((Chem.MolFromSmiles(activated[0]),
                           Chem.MolFromSmiles("NCc1ccccc1")))[0][0]
    Chem.SanitizeMol(p)
    print("SNAr sample product:  ", Chem.MolToSmiles(p))


if __name__ == "__main__":
    main()
