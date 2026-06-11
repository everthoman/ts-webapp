from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog("rdApp.*")
from pathlib import Path

# Protected acid forms
TBOC_ESTER   = Chem.MolFromSmarts("[CX3](=O)OC(C)(C)C")          # tert-butyl ester
BN_ESTER     = Chem.MolFromSmarts("[CX3](=O)OCc1ccccc1")          # benzyl ester
FREE_ACID    = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")

# Boronic ester forms
BPIN         = Chem.MolFromSmarts("[BX3]1OC(C)(C)C(C)(C)O1")      # pinacol boronate
BNEOPENTYL   = Chem.MolFromSmarts("[BX3]1OCC(C)(C)CO1")            # neopentylglycol
BCATECHOL    = Chem.MolFromSmarts("[BX3]1Oc2ccccc2O1")             # catechol boronate
FREE_BORONIC = Chem.MolFromSmarts("[BX3]([OX2H])[OX2H]")           # free boronic acid

counts = {
    "tBu_ester": 0, "Bn_ester": 0,
    "Bpin": 0, "B_neopentyl": 0, "B_catechol": 0,
    "free_acid": 0, "free_boronic": 0,
}

for line in Path("reagents/enamine_rush_EU.smi").open():
    parts = line.split()
    if not parts: continue
    mol = Chem.MolFromSmiles(parts[0])
    if mol is None: continue
    if mol.HasSubstructMatch(TBOC_ESTER):   counts["tBu_ester"] += 1
    if mol.HasSubstructMatch(BN_ESTER):     counts["Bn_ester"] += 1
    if mol.HasSubstructMatch(BPIN):         counts["Bpin"] += 1
    if mol.HasSubstructMatch(BNEOPENTYL):   counts["B_neopentyl"] += 1
    if mol.HasSubstructMatch(BCATECHOL):    counts["B_catechol"] += 1
    if mol.HasSubstructMatch(FREE_ACID):    counts["free_acid"] += 1
    if mol.HasSubstructMatch(FREE_BORONIC): counts["free_boronic"] += 1

total = 44963
print(f"{'Group':20s} {'Count':>7s}  {'%':>5s}")
print("-" * 36)
for k, v in counts.items():
    print(f"{k:20s} {v:7,}  ({100*v/total:.1f}%)")
