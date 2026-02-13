import pandas as pd

fold_dfs = []

for i in range(5):
    preds_df = pd.read_csv(f"../../QSAR GNN/mol_predsF{i}.csv")

    smiles_df = pd.read_csv(f"../toxicidade/toxicidade_fold{i + 1}.csv")

    assert len(preds_df) == len(smiles_df), f"Erro: Fold {i} tem tamanhos diferentes!"

    merged_df = pd.concat([smiles_df, preds_df], axis=1)


    merged_df.to_csv(f"merged_fold{i}.csv", index=False)

    merged_df["fold"] = i

    fold_dfs.append(merged_df)

df_total = pd.concat(fold_dfs, ignore_index=True)

df_total.to_csv("merged_all_folds.csv", index=False)

print(df_total.head())
