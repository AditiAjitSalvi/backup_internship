
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

try:
    df = pd.read_excel(r"c:\Users\aditi\Downloads\Internship\code_\wagone_configuration_for traing.xlsx")
    print("Columns:", df.columns.tolist())
    print("\nFirst Row Transposed:")
    print(df.iloc[0])
except Exception as e:
    print(e)
