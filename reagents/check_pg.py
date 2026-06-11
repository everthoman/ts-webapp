from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from pathlib import Path

BOC  = Chem.MolFromSmarts("[NX3][C](=O)OC(C)(C)C")
CBZ  = Chem.MolFromSmarts("[NX3][C](=O)OCc1ccccc1")
FMOC = Chem.MolFromSmarts("[NX3][C](=O)OCC1c2ccccc2-c2ccccc21")
FREE_AMINE = Chem.MolFromSmarts("[NX3;H1,H2;!$(NC=O);!$(NS)]")

counts = {"boc": 0, "cbz": 0, "fmoc": 0,
          "boc_only": 0, "cbz_only": 0,
          "boc_and_free": 0, "cbz_and_free": 0}
total = 0

for line in Path("reagents/enamine_rush_EU.smi").open():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    smi = line.split()[0]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    total += 1
    has_boc  = mol.HasSubstructMatch(BOC)
    has_cbz  = mol.HasSubstructMatch(CBZ)
    has_fmoc = mol.HasSubstructMatch(FMOC)
    has_free = mol.HasSubstructMatch(FREE_AMINE)
    if has_boc:
        counts["boc"] += 1
        if not has_free: counts["boc_only"] += 1
        else:            counts["boc_and_free"] += 1
    if has_cbz:
        counts["cbz"] += 1
        if not has_free: counts["cbz_only"] += 1
        else:            counts["cbz_and_free"] += 1
    if has_fmoc:
        counts["fmoc"] += 1

print(f"Total: {total:,}")
for k, v in counts.items():
    print(f"  {k:20s} {v:6,}  ({100*v/total:.1f}%)")
