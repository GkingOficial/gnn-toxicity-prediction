import os

pasta_base = '../../Gini_Hung-QSAR_GCN-main/imgF5/Fold 5/'
prefixo = 'Fold5_'
subpastas = ['FN', 'FP', 'TN', 'TP']

for subpasta in subpastas:
    caminho_completo = os.path.join(pasta_base, subpasta)
    
    for nome_arquivo in os.listdir(caminho_completo):
        caminho_antigo = os.path.join(caminho_completo, nome_arquivo)
        
        if os.path.isfile(caminho_antigo):
            novo_nome = prefixo + nome_arquivo
            caminho_novo = os.path.join(caminho_completo, novo_nome)
            os.rename(caminho_antigo, caminho_novo)
