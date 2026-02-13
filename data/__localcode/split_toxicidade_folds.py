import pandas as pd
from sklearn.model_selection import StratifiedKFold

def split_toxicidade_folds(input_file: str, output_prefix: str, n_splits: int = 5):
    df = pd.read_csv(input_file)
    
    X = df['SMILES']
    y = df['Label']
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        df_fold = df.iloc[test_idx]
        output_file = f"../toxicidade/{output_prefix}_fold{fold+1}.csv"
        df_fold.to_csv(output_file, index=False)
        print(f"Fold {fold+1} salvo em: {output_file}")

input_file = "../toxicidade/Toxicidade.csv"
output_prefix = "toxicidade"
split_toxicidade_folds(input_file, output_prefix)