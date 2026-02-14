# import numpy as np
# import os
# import time
# import sys
# import random
# import rdkit
# import tensorflow as tf

# from rdkit import Chem

# from sklearn.metrics import accuracy_score, roc_auc_score,recall_score,matthews_corrcoef
# from rdkit import Chem
# from rdkit import Chem
# from rdkit.Chem.AllChem import Compute2DCoords
# from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit.Chem import rdDepictor

# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole
# from IPython.display import SVG

# from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
# from IPython.display import SVG
# from rdkit import rdBase
# from rdkit.Chem import rdDepictor
# from rdkit.Chem.Draw import rdMolDraw2D

import pandas as pd

# Carregar o arquivo CSV
df = pd.read_csv('novo_arquivo.csv')

df = df.drop(df.columns[[2, 3]], axis=1)

# coluna_6 = df.pop(df.columns[3])
# df.insert(0, coluna_6.name, coluna_6)

df.to_csv('novo_arquivo.csv', index=False)

print(df)


# o1cccc1
#    O C C C C
# O [0 1 0 0 1]
# C [1 0 1 0 0]
# C [0 1 0 1 0]
# C [0 0 1 0 1]
# C [1 0 0 1 0]


# O [1 0 0 0 0]
# C [0 1 0 0 0]
# C [0 0 1 0 0]
# C [0 0 0 1 0]
# C [0 0 0 0 1]

# 5 shape

