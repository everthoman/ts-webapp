from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

# [#6:2][C:1](=O)OX — the #6 neighbour excludes carbamates (where the carbonyl C has N, not C)
rxns = [
    ("Fmoc",     AllChem.ReactionFromSmarts("[N:1][C](=O)OCC1c2ccccc2-c2ccccc21>>[N:1]")),
    ("Boc",      AllChem.ReactionFromSmarts("[N:1][C](=O)OC(C)(C)C>>[N:1]")),
    ("Cbz",      AllChem.ReactionFromSmarts("[N:1][C](=O)OCc1ccccc1>>[N:1]")),
    ("tBu_ester",AllChem.ReactionFromSmarts("[#6:2][C:1](=O)OC(C)(C)C>>[#6:2][C:1](=O)O")),
    ("Bn_ester", AllChem.ReactionFromSmarts("[#6:2][C:1](=O)OCc1ccccc1>>[#6:2][C:1](=O)O")),
    ("Bpin",     AllChem.ReactionFromSmarts("[#6:1][B]1OC(C)(C)C(C)(C)O1>>[#6:1][B](O)O")),
]

def deprotect(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    changed = True
    while changed:
        changed = False
        for name, rxn in rxns:
            products = rxn.RunReactants((mol,))
            if not products:
                continue
            try:
                p = products[0][0]
                Chem.SanitizeMol(p)
                mol = p
                changed = True
                break
            except Exception:
                continue
    return Chem.MolToSmiles(mol)

tests = [
    # (description, input_smi, expected_smi)
    # tBu ester -> acid
    ("tBu ester",         "CC(=O)OC(C)(C)C",                         "CC(=O)O"),
    # Bn ester -> acid
    ("Bn ester",          "CC(=O)OCc1ccccc1",                         "CC(=O)O"),
    # Bpin -> boronic acid
    ("Bpin",              "B1(c2ccccc2)OC(C)(C)C(C)(C)O1",            "OB(O)c1ccccc1"),
    # Boc carbamate must NOT be touched by tBu-ester rule (already handled by Boc rule)
    ("Boc NOT as ester",  "CC(C)(C)OC(=O)NC",                         "CN"),
    # Cbz carbamate must NOT be touched by Bn-ester rule
    ("Cbz NOT as ester",  "O=C(OCc1ccccc1)NC",                        "CN"),
    # Fmoc (already tested, regression check)
    ("Fmoc",              "O=C(OCC1c2ccccc2-c2ccccc21)NC",            "CN"),
    # Mixed: Boc-amino acid tBu ester -> free amino acid
    ("Boc+tBu -> Ala",    "CC(NC(=O)OC(C)(C)C)C(=O)OC(C)(C)C",       "CC(N)C(=O)O"),
    # Aryl Bpin (Suzuki BB) -> aryl boronic acid
    ("Aryl Bpin",         "B1(c2ccncc2)OC(C)(C)C(C)(C)O1",            "OB(O)c1ccncc1"),
    # No PG (unchanged)
    ("no PG",             "c1ccccc1",                                  "c1ccccc1"),
]

all_ok = True
for desc, inp, expected in tests:
    result = deprotect(inp)
    exp_can = Chem.MolToSmiles(Chem.MolFromSmiles(expected))
    ok = result == exp_can
    if not ok:
        all_ok = False
    print(f"{'OK  ' if ok else 'FAIL'}  {desc}: {result}  (expected {exp_can})")

print()
print("All OK" if all_ok else "FAILURES FOUND")
