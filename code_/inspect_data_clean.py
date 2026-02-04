import pandas as pd
import os

base_dir = r"C:\Users\aditi\Downloads\Internship\code_"
seq_file = os.path.join(base_dir, "sequnce for traing.xlsx")
config_file = os.path.join(base_dir, "wagone_configuration_for traing.xlsx")

def print_cols(name, path):
    try:
        df = pd.read_excel(path)
        print(f"--- {name} ---")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print("First row sample:")
        print(df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error reading {name}: {e}")

print_cols("Sequence", seq_file)
print("\n")
print_cols("Config", config_file)
