import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from sklearn.metrics import classification_report

df = pd.read_csv("merged_all_folds.csv")

positivos = df.copy()

def calc_logp(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        return Crippen.MolLogP(mol) if mol else None
    except:
        return None

positivos["logP"] = positivos["SMILES"].apply(calc_logp)

positivos["Reclassificado"] = (positivos["logP"] >= 2.5).astype(int)

df = df.merge(
    positivos[["SMILES", "Reclassificado", "logP"]],
    on="SMILES",
    how="left"
)

df["Ajustado"] = df["Reclassificado"].combine_first(df["Predito"]).astype(int)  

df["logP"] = df["logP"].map(lambda x: f"{x:.4f}".replace('.', ',') if pd.notnull(x) else '')

df.to_csv("Total_logp.csv", index=False)

print("Métricas após reclassificação com logP:")
print(classification_report(df["Real"], df["Ajustado"], digits=4))