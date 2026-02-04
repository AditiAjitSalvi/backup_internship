
import pandas as pd
df = pd.read_excel(r"c:\Users\aditi\Downloads\Internship\code_\wagone_configuration_for traing.xlsx")
print(list(df.columns))
print(df.iloc[0].tolist())
