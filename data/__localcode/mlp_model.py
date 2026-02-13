import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, recall_score, precision_score, f1_score, confusion_matrix

def load_fp_folds():
    folds = []
    for i in range(5):
        df = pd.read_csv(f"../mutagenicity/fp_folds/mutagenicity_fp_fold{i+1}.csv")
        X = df.iloc[:, :2048].values
        y = df["Activity"].values
        folds.append((X, y))
    return folds

folds = load_fp_folds()
results = []
logs = []

for i in range(5):
    print(f"\n===== Fold {i+1}/5 =====")

    X_train = np.vstack([folds[j][0] for j in range(5) if j != i])
    y_train = np.hstack([folds[j][1] for j in range(5) if j != i])
    X_test, y_test = folds[i]

    # MLP
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, verbose = False)
    model.fit(X_train, y_train)
    
    # TRAIN
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_train, y_pred_train).ravel()
    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_pred_train),
        "AUROC": roc_auc_score(y_train, y_prob_train),
        "MCC": matthews_corrcoef(y_train, y_pred_train),
        "Recall": recall_score(y_train, y_pred_train),
        "Specificity": tn_t / (tn_t + fp_t),
        "Precision": precision_score(y_train, y_pred_train),
        "F1-score": f1_score(y_train, y_pred_train),
        "TP": tp_t, "FP": fp_t, "FN": fn_t, "TN": tn_t,
    }

    # TEST
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "AUROC": roc_auc_score(y_test, y_prob_test),
        "MCC": matthews_corrcoef(y_test, y_pred_test),
        "Recall": recall_score(y_test, y_pred_test),
        "Specificity": tn / (tn + fp),
        "Precision": precision_score(y_test, y_pred_test),
        "F1-score": f1_score(y_test, y_pred_test),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }

    logs.append({
        "Fold": i + 1,
        "Treino": train_metrics,
        "Teste": test_metrics
    })

    for idx in range(len(y_test)):
        results.append({
            "Fold": i + 1,
            "Index": idx,
            "Probabilidade": y_prob_test[idx],
            "Predito": y_pred_test[idx],
            "Real": y_test[idx]
        })

df_results = pd.DataFrame(results)
df_results.to_csv("mlp_mutag_preds_all_folds.csv", index=False)

print("\nArquivo mlp_mutag_preds_all_folds.csv salvo com sucesso.")

for log in logs:
    print(f"\n===== Fold {log['Fold']} =====")
    print("Treino:")
    for k, v in log["Treino"].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("Teste:")
    for k, v in log["Teste"].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")