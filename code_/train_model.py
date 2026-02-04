
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Paths
BASE_DIR = r"C:\Users\aditi\Downloads\Internship\code_"
SEQ_FILE = os.path.join(BASE_DIR, "sequnce for traing.xlsx")
CONFIG_FILE = os.path.join(BASE_DIR, "wagone_configuration_for traing.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    print("⏳ Loading data...")
    try:
        df_seq = pd.read_excel(SEQ_FILE)
        df_config = pd.read_excel(CONFIG_FILE)
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return

    # Clean Column Names
    df_seq.columns = [c.strip() for c in df_seq.columns]
    df_config.columns = [c.strip() for c in df_config.columns]

    # --- MERGE DATA ---
    # Config has 'ProjectID' and 'ID'. Sequence has 'project_id'.
    # We will try to map Config['ProjectID'] -> Seq['project_id']
    
    if 'ProjectID' in df_config.columns:
        print("   Using Config 'ProjectID' for merge")
        df_config.rename(columns={'ProjectID': 'project_id'}, inplace=True)
    elif 'ID' in df_config.columns:
         print("   Using Config 'ID' as project_id (Fallback)")
         df_config.rename(columns={'ID': 'project_id'}, inplace=True)
         
    if 'project_id' not in df_seq.columns or 'project_id' not in df_config.columns:
         print("❌ Missing 'project_id' key in one of the files.")
         print(f"   Seq Keys: {df_seq.columns.tolist()}")
         print(f"   Config Keys: {df_config.columns.tolist()}")
         return

    print("🔗 Merging...")
    df = pd.merge(df_seq, df_config, on='project_id', how='inner')
    print(f"   Merged shape: {df.shape}")
    
    # --- FEATURES ---
    # Based on verify_cols output:
    feature_candidates = [
        'SuperfastSpeed', 'FastSpeed', 'SlowSpeed', 
        'LiftSpeed', 'LowerDownSpeed', 'LiftStrokeSpeed', # Looks like Speed not Time
        'WagonMinStationNo', 'WagonMaxStationNo', 'BasicPosition',
        'station_no' # Helper from sequence?
    ]
    
    valid_features = [c for c in feature_candidates if c in df.columns]
    # Check if 'station_no' is in sequence or config?
    if 'station_no' in df.columns:
        valid_features.append('station_no')
    
    print(f"   Training Features: {valid_features}")
    

    # --- TARGETS ---
    # We want to learn 'Duration' of a step (Physics), not accumulated time.
    # We calculate delta info from 'total_time_sec' if 'duration' not present.
    
    # Sort by Project, Wagon, Step (assuming imported order is somewhat sequential or use step_no)
    if 'step_no' in df.columns:
        df = df.sort_values(by=['project_id', 'wagon_no', 'step_no'])
    else:
        df = df.sort_values(by=['project_id', 'wagon_no'])
        
    target_col = 'duration'
    if 'duration' not in df.columns:
        if 'total_time_sec' in df.columns:
            print("   Calculated 'duration' from 'total_time_sec' delta.")
            # Group by Wagon and Diff
            df['duration'] = df.groupby(['project_id', 'wagon_no'])['total_time_sec'].diff().fillna(df['total_time_sec']) 
            # First item might be large if it starts late? 
            # Or if it's the first step, total_time is the duration.
            # However, if Wagon starts at t=0, first duration ok.
            # Handle negatives? sequence should be increasing.
            df['duration'] = df['duration'].clip(lower=0)
        else:
             # Fallback
             print("⚠️  No time column found for duration. Using fake target.")
             df['duration'] = 100 # Dummy
    
    target_time = 'duration'

    # Encode Command
    le_cmd = LabelEncoder()
    if 'command' in df.columns:
        df['command_encoded'] = le_cmd.fit_transform(df['command'].astype(str))
        valid_features.append('command_encoded')
        
    X = df[valid_features].fillna(0)
    y_time = df[target_time].fillna(0)
    
    # Train Time Model (Duration)
    print(f"🤖 Training Random Forest Regressor on '{target_time}'...")
    rf_time = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_time.fit(X, y_time)
    
    joblib.dump(rf_time, os.path.join(MODEL_DIR, "rf_time_model.pkl"))
    joblib.dump(valid_features, os.path.join(MODEL_DIR, "model_features.pkl"))
    if 'command' in df.columns:
        joblib.dump(le_cmd, os.path.join(MODEL_DIR, "le_command.pkl"))
        
    print("✅ Training Complete.")
    print(f"   MSE: {mean_squared_error(y_time, rf_time.predict(X)):.4f}")

if __name__ == "__main__":
    train()
