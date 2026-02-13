from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd

def smiles_to_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:  
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        return np.array(fp)
    else:
        return np.zeros(n_bits)

df = pd.read_csv("../toxicidade/Toxicidade.csv")



X = np.array([smiles_to_fingerprint(smi) for smi in df['SMILES']])
y = df['Label'].values

df_fp = pd.DataFrame(X)
df_fp['Label'] = y

df_fp.to_csv("../toxicidade/toxicidade_fp.csv", index=False)
print("Novo dataset salvo como 'toxicidade_fp.csv' 🚀")

print(f"Dimensão de X: {X.shape}, Dimensão de y: {y.shape}")
