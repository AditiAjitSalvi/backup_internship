import pandas as pd
import os

base_dir = r"C:\Users\aditi\Downloads\Internship\code_"
seq_file = os.path.join(base_dir, "sequnce for traing.xlsx")
config_file = os.path.join(base_dir, "wagone_configuration_for traing.xlsx")

print(f"Reading {seq_file}...")
try:
    df_seq = pd.read_excel(seq_file)
    print("Columns:", df_seq.columns.tolist())
    print("First 3 rows:")
    print(df_seq.head(3).to_string())
except Exception as e:
    print(f"Error reading seq file: {e}")

print(f"\nReading {config_file}...")
try:
    df_config = pd.read_excel(config_file)
    print("Columns:", df_config.columns.tolist())
    print("First 3 rows:")
    print(df_config.head(3).to_string())
except Exception as e:
    print(f"Error reading config file: {e}")
