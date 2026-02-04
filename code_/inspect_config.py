
import pandas as pd
try:
    df = pd.read_csv(r"c:\Users\aditi\Downloads\Internship\Internship\code\wagon_config.csv")
    print("Columns:", df.columns.tolist())
    print("First row:", df.iloc[0].to_dict())
except Exception as e:
    print(e)
