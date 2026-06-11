from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from pathlib import Path

sdf = Path("/opt/webapps/TS/reagents/Enamine_Rush-Delivery_Building_Blocks-EU_44963cmpd_20260602.sdf")
out = Path("/opt/webapps/TS/reagents/enamine_rush_EU.smi")
n_ok = n_bad = 0
with open(out, "w") as fh:
    for mol in Chem.SDMolSupplier(str(sdf)):
        if mol is None:
            n_bad += 1
            continue
        name = mol.GetProp("Catalog_ID") if mol.HasProp("Catalog_ID") else f"mol_{n_ok}"
        smi = Chem.MolToSmiles(mol)
        fh.write(f"{smi} {name}\n")
        n_ok += 1
print(f"Done: {n_ok} ok, {n_bad} bad -> {out}")
