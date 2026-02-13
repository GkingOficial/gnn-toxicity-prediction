from sklearn.metrics import roc_auc_score
import pandas as pd

df = pd.read_csv("merged_all_folds.csv")

auc_total = roc_auc_score(df["Real"], df["Probabilidade"])
print("AUC total combinado:", round(auc_total, 4))
