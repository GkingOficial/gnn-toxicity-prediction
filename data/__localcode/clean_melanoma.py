import pandas as pd

def clean_melanoma_dataset(input_file: str, output_file: str):
    df = pd.read_csv(input_file)
    
    df['pIC50'] = df['pIC50'].str.replace(',', '.').astype(float)
    df['pIC50'] = (df['pIC50'] > 5.0).astype(int)
    
    df.to_csv(output_file, index=False)
    print(f"Base de dados tratada e salva em: {output_file}")

input_file = "../melanoma/SMILES e pIC50 melanoma.csv"
output_file = "../melanoma/melanoma_cleaned.csv"
clean_melanoma_dataset(input_file, output_file)