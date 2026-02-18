import csv
from tabulate import tabulate
from collections import defaultdict

def generate_dip_time_table(station_csv, sequence_csv, max_cycle_time=600):
    """
    Generates a Dip Time Table by merging station data and sequence instructions.
    
    Logic:
    1. Load station metadata from test.csv.
    2. Collect all "PUT ON" (Dip In) and "GET FROM" (Dip Out) events from sequence.csv.
    3. Pair events per station:
       - Normal: IN followed by OUT.
       - Cyclic: IN at end of sequence followed by OUT at start (OUT < IN).
       - Single-point: Only IN (Unloading) or only OUT (Loading).
    """
    
    # 1. Load Station Master Data (test.csv)
    stations = {}
    try:
        with open(station_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stn_id = row.get('Station No')
                if stn_id:
                    # Capture distance using multiple potential key names
                    dist = row.get('Distance(mm)') or row.get('distance_mm') or row.get('Distance') or '0'
                    stations[stn_id] = {
                        'Process Name': row.get('Process Name', ''),
                        'Criticality': row.get('Criticality', 'Low'),
                        'Distance': dist
                    }
    except Exception as e:
        print(f"Error reading station file: {e}")
        return []

    # 2. Collect all actions per station
    station_actions = defaultdict(list)
    try:
        with open(sequence_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                wagon = row.get('Wagon No')
                instr = row.get('Instruction', '').upper().strip()
                val = row.get('Instruction Value', '').strip()
                try:
                    time_val = float(row.get('Time', 0))
                except:
                    time_val = 0.0
                
                if instr == "PUT ON" or "PUT" in instr:
                    station_actions[val].append({'type': 'IN', 'wagon': wagon, 'time': time_val})
                elif instr == "GET FROM" or "GET" in instr:
                    station_actions[val].append({'type': 'OUT', 'wagon': wagon, 'time': time_val})
    except Exception as e:
        print(f"Error reading sequence file: {e}")
        return []

    # 3. Pair events and handle special cases
    final_pairs = []
    
    # Determine all unique stations involved
    all_involved_stns = sorted(station_actions.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    for stn_id in all_involved_stns:
        actions = sorted(station_actions[stn_id], key=lambda x: x['time'])
        ins = [a for a in actions if a['type'] == 'IN']
        outs = [a for a in actions if a['type'] == 'OUT']
        
        used_ins = set()
        used_outs = set()
        
        # A. Normal Pairing (IN then OUT)
        for i, in_act in enumerate(ins):
            for j, out_act in enumerate(outs):
                if j not in used_outs and out_act['time'] >= in_act['time']:
                    final_pairs.append({
                        'wagon': in_act['wagon'], # Typically follows the wagon that put it in
                        'stn': stn_id,
                        'dip_in': in_act['time'],
                        'dip_out': out_act['time'],
                        'type': 'Normal'
                    })
                    used_ins.add(i)
                    used_outs.add(j)
                    break
        
        # B. Cyclic Pairing (OUT at start, IN at end)
        # Requirement: "If DipOutTime < DipInTime: DipOutTime += max_cycle_time"
        for i, in_act in enumerate(ins):
            if i in used_ins: continue
            for j, out_act in enumerate(outs):
                if j not in used_outs and out_act['time'] < in_act['time']:
                    final_pairs.append({
                        'wagon': in_act['wagon'], 
                        'stn': stn_id,
                        'dip_in': in_act['time'],
                        'dip_out': out_act['time'] + max_cycle_time,
                        'type': 'Cyclic'
                    })
                    used_ins.add(i)
                    used_outs.add(j)
                    break
                    
        # C. Single Point - Only OUT (Example: Station 1 Loading)
        for j, out_act in enumerate(outs):
            if j not in used_outs:
                final_pairs.append({
                    'wagon': out_act['wagon'],
                    'stn': stn_id,
                    'dip_in': 0.0, # Assume start of observation for loading/residual
                    'dip_out': out_act['time'],
                    'type': 'Residual OUT'
                })
                used_outs.add(j)
                
        # D. Single Point - Only IN (Example: Station 12 Unloading)
        for i, in_act in enumerate(ins):
            if i not in used_ins:
                final_pairs.append({
                    'wagon': in_act['wagon'],
                    'stn': stn_id,
                    'dip_in': in_act['time'],
                    'dip_out': in_act['time'], # Zero Dip Time for final unload point? Or duration
                    'type': 'Residual IN'
                })
                used_ins.add(i)

    # 4. Final Formatting
    final_table = []
    # Sort results by Wagon then DipIn for a clean chronological view
    final_pairs.sort(key=lambda x: (int(x['wagon']) if x['wagon'].isdigit() else 0, x['dip_in']))
    
    for res in final_pairs:
        stn_meta = stations.get(res['stn'], {})
        dip_time = res['dip_out'] - res['dip_in']
        
        final_table.append({
            "Wagon": res['wagon'],
            "Station": res['stn'],
            "Process Name": stn_meta.get('Process Name', 'Unknown'),
            "Criticality": stn_meta.get('Criticality', 'Low'),
            "DipIn": res['dip_in'],
            "DipOut": res['dip_out'],
            "DipTime": dip_time,
            "Distance": stn_meta.get('Distance', '0')
        })
        
    return final_table

def display_results(data):
    """Prints the final table using tabulate with the requested columns."""
    if not data:
        print("No data to display.")
        return
        
    headers = ["Wagon", "Station", "Process Name", "Criticality", "DipIn", "DipOut", "DipTime", "Distance"]
    rows = []
    for d in data:
        rows.append([
            d["Wagon"],
            d["Station"],
            d["Process Name"],
            d["Criticality"],
            d["DipIn"],
            d["DipOut"],
            d["DipTime"],
            d["Distance"]
        ])
        
    print("\nINDUSTRIAL DIP TIME CALCULATION TABLE")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    stations_path = "testing_files/test2.csv"
    sequence_path = "testing_files/sequnce2.csv"
    
    # We'll use 600 as the cycle time for overflow logic, 
    # as the sequence goes up to ~520 seconds.
    table_data = generate_dip_time_table(stations_path, sequence_path, max_cycle_time=600)
    display_results(table_data)
