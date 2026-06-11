from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

rxns = [
    AllChem.ReactionFromSmarts("[N:1][C](=O)OCC1c2ccccc2-c2ccccc21>>[N:1]"),  # Fmoc
    AllChem.ReactionFromSmarts("[N:1][C](=O)OC(C)(C)C>>[N:1]"),               # Boc
    AllChem.ReactionFromSmarts("[N:1][C](=O)OCc1ccccc1>>[N:1]"),              # Cbz
]

def deprotect(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    changed = True
    while changed:
        changed = False
        for rxn in rxns:
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
    ("Fmoc-Gly",      "NCC(=O)O",   "O=C(O)CNC(=O)OCC1c2ccccc2-c2ccccc21"),
    ("Boc-piperazine", "C1CNCCN1",  "CC(C)(C)OC(=O)N1CCNCC1"),
    ("Cbz-amine",      "NCc1ccccc1", "O=C(OCc1ccccc1)NCc1ccccc1"),
    ("no PG",          "c1ccccc1",   "c1ccccc1"),
]

for name, expected, protected in tests:
    result = deprotect(protected)
    exp_can = Chem.MolToSmiles(Chem.MolFromSmiles(expected))
    status = "OK  " if result == exp_can else "FAIL"
    print(f"{status}  {name}: -> {result}  (expected {exp_can})")
