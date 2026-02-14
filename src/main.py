import numpy as np
import tensorflow as tf
from utils.utils import load_folds
from models.mc_dropout import mc_dropout
from training.train_gcn import training
from config import TrainConfig

def main():
  FLAGS = TrainConfig()

  folds = load_folds()
  model_name = "Refactory1402"

  for fold_index in range(1): # len(folds)
    print(f"\n===== Fold {fold_index + 1}/5 =====")
    smi_train = []
    prop_train = []

    for j, (smi, prop) in enumerate(folds):
      if j == fold_index:
        smi_test = smi
        prop_test = prop
      else:
        smi_train.extend(smi)
        prop_train.extend(prop)

    smi_train = np.array(smi_train)
    prop_train = np.array(prop_train)
    smi_test = np.array(smi_test)
    prop_test = np.array(prop_test)

    unique, counts = np.unique(prop_train, return_counts=True)
    print("Treino:", dict(zip(unique, counts)))

    unique, counts = np.unique(prop_test, return_counts=True)
    print("Teste:", dict(zip(unique, counts)))

    tf.compat.v1.reset_default_graph()

    model = mc_dropout(FLAGS)
    
    training(
      model,
      FLAGS,
      f"{model_name}_fold{fold_index}",
      smi_train,
      prop_train,
      smi_test,
      prop_test,
      fold_index
    )

if __name__ == "__main__":
  main()