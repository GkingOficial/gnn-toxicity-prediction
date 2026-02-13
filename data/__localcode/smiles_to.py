import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os

def smiles_to_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)

input_dir = "../melanoma"
output_dir = "../melanoma/fp_folds"
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 6):
    input_path = f"{input_dir}/melanoma_fold{i}.csv"
    output_path = f"{output_dir}/melanoma_fp_fold{i}.csv"

    df = pd.read_csv(input_path)
    print(f"🔄 Processando fold {i}...")

    X = np.array([smiles_to_fingerprint(smi) for smi in df['smiles']])
    y = df['pIC50'].values

    df_fp = pd.DataFrame(X)
    df_fp['pIC50'] = y

    df_fp.to_csv(output_path, index=False)
    print(f"✅ Fold {i} salvo como {output_path}")
