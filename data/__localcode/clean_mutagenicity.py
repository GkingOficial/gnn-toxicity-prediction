import pandas as pd

def clean_mutagenicity_dataset(input_file: str, output_file: str):
    df = pd.read_csv(input_file)

    df_cleaned = df[['Canonical_Smiles', 'Activity']]
    
    df_cleaned.to_csv(output_file, index=False)
    print(f"Base de dados tratada e salva em: {output_file}")

input_file = "../Mutagenicity_N6512.csv"
output_file = "../mutagenicity_cleaned.csv"

clean_mutagenicity_dataset(input_file, output_file)
