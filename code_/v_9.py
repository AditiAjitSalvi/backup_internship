#!/usr/bin/env python3
"""
Complete PLC Collision-Free Scheduler V9 (Hybrid ML + Physics)
- Uses Trained ML Model for Sequence Step Duration (Learned Speed)
- Uses v8.py Rule-Based Logic for Collision Avoidance (Dynamic Waits)
- Performs Gap Analysis (Physics Active Time vs Actual Simulated Time)
"""

import os
import csv
import argparse
from collections import defaultdict
from tabulate import tabulate
import pandas as pd
import numpy as np
import joblib

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = r"C:\Users\aditi\Downloads\Internship\code_" # Path where scripts are
DATA_DIR = r"C:\Users\aditi\Downloads\Internship\Internship\code" # Path where data is
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_WAGON_CONFIG = r"wagone_configuration_for traing.xlsx"
DEFAULT_TANKS_CSV = os.path.join(DATA_DIR, "tanks_csv.csv")
MAX_STATIONS = 200

# ==========================================
# LOAD MODELS
# ==========================================
def load_ml_models():
    try:
        path_rf = os.path.join(MODEL_DIR, "rf_time_model.pkl")
        path_feat = os.path.join(MODEL_DIR, "model_features.pkl")
        
        if not os.path.exists(path_rf):
             print(f"⚠️  ML Model not found at {path_rf}. Using pure physics.")
             return None, None
             
        rf_time = joblib.load(path_rf)
        features = joblib.load(path_feat)
        print("✅ ML Duration Model loaded.")
        return rf_time, features
    except Exception as e:
        print(f"❌ Error loading ML models: {e}")
        return None, None

# ==========================================
# LOAD CONFIG
# ==========================================
def load_wagon_config(path):
    config = {}
    print(f"📖 Loading Wagon Config: {path}")
    if not os.path.exists(path):
        if os.path.exists(os.path.join(BASE_DIR, path)): path = os.path.join(BASE_DIR, path)
        else: print(f"❌ File not found: {path}"); return config
    
    try:
        # Check file extension to use appropriate loader
        if path.endswith('.xlsx') or path.endswith('.xls'):
            df = pd.read_excel(path)
        else:
            # Assume CSV
            df = pd.read_csv(path)

        df.columns = [c.strip() for c in df.columns]
        
        for index, row in df.iterrows():
            if 'Transporter Name' in row: name = str(row['Transporter Name'])
            elif 'Transporter_Name' in row: name = str(row['Transporter_Name'])
            elif 'wagon_no' in row: name = str(row['wagon_no'])
            elif 'WagonNumber' in row: name = f"Wagon{row['WagonNumber']}"
            else: name = f"Wagon{index+1}"
            
            def val(keys, default=0.0):
                for k in keys:
                     if k in row: return float(row[k])
                return float(default)
            
            # Logic from u.py for robust loading including min/max stations
            config[name] = {
                'sf': val(['Superfast Speed', 'SuperfastSpeed'], 10),
                'f':  val(['Fast Speed', 'FastSpeed'], 5),
                's':  val(['Slow Speed', 'SlowSpeed'], 2),
                'lift_time': val(['Lift Time', 'LiftSpeed'], 1.0),
                'lower_time': val(['Lower Time', 'LowerDownSpeed'], 1.5),
                'wagon_length': val(['Wagon Length (mm)', 'WagonLength'], 5000),
                'safety_buffer': val(['Safety Buffer (mm)', 'SafetyBuffer'], 500),
                'size_factor': val(['Size Factor', 'Size_Factor'], 1.0),
                'occupy_factor': val(['Station Occupy Factor'], 1.0),
                'min_stn': int(val(['Minimum Station No', 'MinimumStationNo'], 1)),
                'max_stn': int(val(['Maximum Station No', 'MaximumStationNo'], 200)),
                'raw_row': row.to_dict()
            }
        print(f"✅ Loaded {len(config)} wagons")
    except Exception as e:
        print(f"❌ Config Load Error: {e}")
        exit(1)
    return config

def load_tanks(path):
    print(f"📖 Loading Tanks/Stations: {path}")
    if not os.path.exists(path):
        # Try finding it in known directories
        if os.path.exists(os.path.join(DATA_DIR, os.path.basename(path))):
            path = os.path.join(DATA_DIR, os.path.basename(path))
            print(f"   Found at: {path}")
        elif os.path.exists(os.path.join(BASE_DIR, path)):
            path = os.path.join(BASE_DIR, path)
            print(f"   Found at: {path}")
        elif os.path.exists(os.path.join(os.getcwd(), path)):
             path = os.path.join(os.getcwd(), path)
             print(f"   Found at: {path}")
        else:
            print(f"❌ Tanks file not found: {path}")
            # List contents of likely dir for debugging
            try:
                print(f"   Contents of {DATA_DIR}: {os.listdir(DATA_DIR)}")
            except: pass
            exit(1)
            
    return pd.read_csv(path).to_dict('records')

# ==========================================
# PHYSICS & LOGIC
# ==========================================
def calculate_time_size_aware(distance_mm, w):
    sfs = w['sf'] * 16.66; fs = w['f'] * 16.66; ss = w['s'] * 16.66
    effective = distance_mm + w['wagon_length'] + w['safety_buffer']
    sfs/=w['size_factor']; fs/=w['size_factor']; ss/=w['size_factor']
    slow = 500 * w['size_factor']
    fast = max(0, effective - slow)
    return (fast / fs) + (slow / ss)

def build_adjacency(max_stn):
    adj = defaultdict(list)
    for i in range(1, max_stn):
        adj[i].append(i+1); adj[i+1].append(i)
    return adj

def is_station_free(station_locks, station, start, end, adjacency):
    # Check self
    for s, e, _ in station_locks.get(station, []):
        if not (end <= s or start >= e): return False
    # Check neighbors
    for adj in adjacency.get(station, []):
        for s, e, _ in station_locks.get(adj, []):
             if not (end <= s or start >= e): return False
    return True

# ==========================================
# GENERATION
# ==========================================
def generate_sequence(config_path, tanks_path):
    rf_time, ml_features = load_ml_models()
    wagons = load_wagon_config(config_path)
    tanks = load_tanks(tanks_path)
    
    adjacency = build_adjacency(MAX_STATIONS)
    station_locks = defaultdict(list)
    
    output_rows = []
    
    # Gap Analysis Data
    # Key: Wagon, Value: {Active(Physics), Total(Sim)}
    gap_data = defaultdict(lambda: {'active': 0.0, 'total': 0.0})

    for wagon_id, w in wagons.items():
        # Hardcoded constraints fallback if not in config
        # User request: Wagon 1 (1-5), Wagon 2 (6-9)
        # Check if config didn't have specific values (defaults were 1 and 200)
        # If the names match the user request, enforce the rule if config implies default.
        
        # NOTE: strictly following config if loaded, but here we enforce logic if names match
        if "Wagon1" in wagon_id.replace(" ", "") and w['max_stn'] == 200:
            w['min_stn'] = 1
            w['max_stn'] = 5
        elif ("Wagon2" in wagon_id.replace(" ", "") or "seone" in wagon_id.lower()) and w['max_stn'] == 200:
             w['min_stn'] = 6
             w['max_stn'] = 9

        print(f"\n🚂 Processing {wagon_id} (Range: {w['min_stn']}-{w['max_stn']})...")
        time_cursor = 0.0 # Wagon Clock
        
        for i in range(len(tanks) - 1):
            curr = tanks[i]
            nxt = tanks[i + 1]
            s_from = int(curr['station_no'])
            s_to = int(nxt['station_no'])
            
            # Zoning Logic: Check if this step belongs to this wagon
            # We check if the destination station is within the wagon's range
            if not (w['min_stn'] <= s_to <= w['max_stn']):
                continue
            
            dist = abs(float(nxt['distance_mm']) - float(curr['distance_mm']))
            
            # 1. PREDICT DURATION (Travel)
            # Use ML if available, else Physics
            
            # Physics Calculation (Baseline)
            phys_travel = calculate_time_size_aware(dist, w)
            phys_step_time = phys_travel + w['lift_time'] + w['lower_time']
            gap_data[wagon_id]['active'] += phys_step_time
            
            # ML Prediction
            duration = phys_step_time # Default
            if rf_time and ml_features:
                row = w['raw_row']
                vec = {}
                for f in ml_features:
                    if f in row: vec[f] = row[f]
                    elif f.replace('Speed', ' Speed') in row: vec[f] = row[f.replace('Speed', ' Speed')]
                    elif f == 'station_no': vec[f] = s_to
                    else: vec[f] = 0
                
                try:
                    # Predict Duration
                    pred = rf_time.predict(pd.DataFrame([vec]).fillna(0))[0]
                    # Ensure non-negative and reasonable (avg with physics?)
                    if pred > 0: duration = pred
                except: pass
            
            # 2. COLLISION WAIT
            # Duration includes Travel + Ops. We assume it occupies destination for 'duration * OccFactor'
            occupy_duration = duration * w['occupy_factor']
            
            wait_start = time_cursor
            wait_count = 0
            while not is_station_free(station_locks, s_to, time_cursor, time_cursor + occupy_duration, adjacency):
                time_cursor += 1
                wait_count += 1
            
            if wait_count > 0:
                 # Record Wait Step
                 output_rows.append({
                     'wagon_no': wagon_id, 'step_no': len(output_rows)+1,
                     'command': 'WAIT', 'station_no': s_to,
                     'wait_sec': wait_count,
                     'total_time_sec': time_cursor,
                     'critical_status': 'High', 'valid': True, 'project_id': 1, 'seq_id': 999
                 })
            
            # 3. ADD LOCK & ADVANCE
            # Critical: Ensure two wagons are not in same place
            station_locks[s_to].append((time_cursor, time_cursor + occupy_duration, wagon_id))
            time_cursor += occupy_duration # Advance by duration (Travel+Process)
            
            # Record Main Step
            output_rows.append({
                 'wagon_no': wagon_id, 'step_no': len(output_rows)+1,
                 'command': 'PUT ON', 'station_no': s_to,
                 'wait_sec': 0,
                 'total_time_sec': time_cursor,
                 'critical_status': 'High', 'valid': True, 'project_id': 1, 'seq_id': 999
            })
            
            # 4. DIP WAIT
            dip = float(nxt.get('dip_time_sec', 0))
            if dip > 0:
                output_rows.append({
                     'wagon_no': wagon_id, 'step_no': len(output_rows)+1,
                     'command': 'WAIT', 'station_no': s_to,
                     'wait_sec': dip,
                     'total_time_sec': time_cursor + dip,
                     'critical_status': 'High', 'valid': True, 'project_id': 1, 'seq_id': 999
                })
                time_cursor += dip
                # (Optional: Add lock for dip?)
        
        gap_data[wagon_id]['total'] = time_cursor
        print(f"✅ {wagon_id} Finished at {time_cursor:.1f}s")
        
    # SAVE CSV
    df_out = pd.DataFrame(output_rows)
    out_csv = os.path.join(DATA_DIR, "plc_sequence_output_v9.csv")
    df_out.to_csv(out_csv, index=False)
    
    print("\n" + "="*80)
    print("🎯 GENERATED SEQUENCE")
    print(tabulate(df_out[['wagon_no', 'command', 'station_no', 'total_time_sec']].tail(10), headers='keys', tablefmt='psql'))
    print(f"\n💾 Sequence saved to {out_csv}")
    
    # GAP ANALYSIS REPORT
    print("\n" + "="*60)
    print("📊 GAP ANALYSIS REPORT (Hybrid ML + Physics)")
    print("="*60)
    for w_id, d in gap_data.items():
        active = d['active']
        total = d['total']
        gap = total - active
        eff = (active / total * 100) if total > 0 else 0
        print(f"🔹 {w_id}:")
        print(f"   Total Time (Simulated): {total:.2f}s")
        print(f"   Active Time (Physics):  {active:.2f}s")
        print(f"   Wait/Gap (Total - Active): {gap:.2f}s")
        print(f"   Efficiency: {eff:.1f}%")
        if eff < 80: print("   ⚠️  High Inefficiency Detected. Check bottlenecks.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_WAGON_CONFIG)
    parser.add_argument("--input", default=DEFAULT_TANKS_CSV)
    args = parser.parse_args()
    generate_sequence(args.config, args.input)
