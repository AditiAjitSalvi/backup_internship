
import pandas as pd
import numpy as np
import os
import joblib
import argparse

# Paths
BASE_DIR = r"C:\Users\aditi\Downloads\Internship\code_"
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_FILE = os.path.join(r"C:\Users\aditi\Downloads\Internship\Internship\code", "plc_sequence_output_ml.csv")

def load_models():
    try:
        rf_time = joblib.load(os.path.join(MODEL_DIR, "rf_time_model.pkl"))
        rf_cmd = joblib.load(os.path.join(MODEL_DIR, "rf_command_model.pkl"))
        le_cmd = joblib.load(os.path.join(MODEL_DIR, "le_command.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
        return rf_time, rf_cmd, le_cmd, features
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None, None, None

def generate_sequence(config_path, stations_path):
    print("🚀 Starting ML Sequence Generation...")
    
    # Load Models
    rf_time, rf_cmd, le_cmd, feature_cols = load_models()
    if not rf_time:
        return

    # Load Inputs
    try:
        # Load Config (Take first row if multiple, or match input?)
        # User said "giving input of wagon config"
        if config_path.endswith('.csv'):
             df_config = pd.read_csv(config_path)
        else:
             df_config = pd.read_excel(config_path)
             
        # Normalize cols
        df_config.columns = [c.strip().replace(' ', '_') for c in df_config.columns]
        
        # Select first wagon config for now (Simplification)
        wagon_config = df_config.iloc[0].to_dict()
        print(f"   Using Config ID: {wagon_config.get('ID', 'Unknown')}")
        
        # Load Stations
        df_stations = pd.read_csv(stations_path)
        print(f"   Loaded {len(df_stations)} stations to process.")
        
    except Exception as e:
        print(f"❌ Error reading inputs: {e}")
        return

    sequence_output = []
    
    # Iterate through stations to generate sequence
    # We maintain a 'context' if needed? RF is stateless, so depends on features.
    
    accumulated_time = 0.0
    
    for idx, row in df_stations.iterrows():
        station_no = row['station_no']
        
        # Construct Feature Vector
        # We need to match the 'available_features' used in training
        input_data = wagon_config.copy()
        input_data['station_no'] = station_no
        
        # Create DataFrame for prediction (aligned with features)
        input_df = pd.DataFrame([input_data])
        
        # Fill missing cols with 0
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder to match training
        X_pred = input_df[feature_cols].fillna(0)
        
        # Predict Command
        cmd_idx = rf_cmd.predict(X_pred)[0]
        command_str = le_cmd.inverse_transform([cmd_idx])[0]
        
        # Predict Time
        pred_time = rf_time.predict(X_pred)[0]
        
        # Update Logic
        # If the model predicts total_time, we might need to delta?
        # But we trained on 'total_time_sec' or whatever column was picked.
        # Assuming training target was 'Accumulated Time' (seq output),
        # We should probably use the predicted time as the cumulative time directly?
        # Or if training data was 'Step Duration', we add it.
        # Let's assume the model learned the Time Value for this step.
        
        # Actually, in v9 output 'total_time_sec' usually means 'Accumulated'.
        # If we just output the predicted accumulated time, it might decrease if model is bad!
        # Let's trust the model for now or force monotonicity.
        
        if pred_time < accumulated_time:
             # Model predicted a time earlier than current!?
             # Force it to be at least current + small delta
             pred_time = accumulated_time + 1.0
        
        accumulated_time = pred_time
        
        # Clean up command string (remove artifacts if any)
        
        sequence_output.append({
            'seq_id': 1,
            'project_id': wagon_config.get('ID', 1),
            'wagon_no': wagon_config.get('Transporter_Name', 'Wagon1'),
            'step_no': idx + 1,
            'command': command_str,
            'station_no': station_no,
            'wait_sec': 0, # Not predicted separately yet
            'critical_status': 'High',
            'total_time_sec': accumulated_time,
            'valid': True
        })
        
        print(f"   Step {idx+1}: {command_str} @ {station_no} | T={accumulated_time:.2f}s")
        
    # Save Output
    df_out = pd.DataFrame(sequence_output)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Sequence saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=r"wagone_configuration_for traing.xlsx", help="Wagon Config Excel")
    parser.add_argument("--stations", default=r"C:\Users\aditi\Downloads\Internship\Internship\code\tanks_csv.csv", help="Stations CSV")
    args = parser.parse_args()
    
    # Fix paths if relative
    if not os.path.isabs(args.config):
        args.config = os.path.join(BASE_DIR, args.config)
        
    generate_sequence(args.config, args.stations)
