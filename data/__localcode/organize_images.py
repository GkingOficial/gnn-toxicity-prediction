import os
import shutil
import pandas as pd

caminho_csv = '../../Gini_Hung-QSAR_GCN-main/imgF5/Fold 5/dict.txt'
pasta_imagens = '../../Gini_Hung-QSAR_GCN-main/imgF5/Fold 5/'
pasta_saida = '../../Gini_Hung-QSAR_GCN-main/imgF5/Fold 5/' 

classes = ['FP', 'FN', 'TP', 'TN']
for classe in classes:
    os.makedirs(os.path.join(pasta_saida, classe), exist_ok=True)

df = pd.read_csv(caminho_csv)

for _, linha in df.iterrows():
    nome_arquivo = linha['Arquivo']
    predito = linha['Predito']
    real = linha['Real']

    if predito == 1 and real == 1:
        destino = 'TP'
    elif predito == 0 and real == 0:
        destino = 'TN'
    elif predito == 1 and real == 0:
        destino = 'FP'
    elif predito == 0 and real == 1:
        destino = 'FN'
    else:
        continue

    origem = os.path.join(pasta_imagens, nome_arquivo)
    destino_final = os.path.join(pasta_saida, destino, nome_arquivo)

    if os.path.exists(origem):
        shutil.copy2(origem, destino_final)
    else:
        print(f'Arquivo não encontrado: {origem}')
