import csv
import json
from diptime.diptime_calculation import Wagon, SequenceProcessor
from tabulate import tabulate

def process_sequence_csv(seq_path, tank_path):
    # 1. Load Station-to-Distance Mapping
    stn_map = {}
    with open(tank_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stn_map[int(row['station_no'])] = float(row['distance_mm'])
            
    # 2. Group instructions by Wagon
    wagon_sequences = {}
    with open(seq_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w_no = row['Wagon No']
            if w_no not in wagon_sequences:
                wagon_sequences[w_no] = []
            
            instr = row['Instruction']
            val = float(row['Instruction Value'])
            
            # Map station number to distance for movement
            if instr.upper() in ["GET FROM", "PUT ON"]:
                if int(val) in stn_map:
                    val_dist = stn_map[int(val)]
                else:
                    val_dist = val # Fallback
            else:
                val_dist = val
                
            wagon_sequences[w_no].append((instr, val_dist, int(val) if instr.upper() in ["GET FROM", "PUT ON"] else None))

    # 3. Process each wagon
    all_dip_data = {} # station_no -> {DipInTime, DipOutTime}
    
    wagon_params = {
        "BasicPosition": 0,
        "SuperfastSpeed": 30,
        "FastSpeed": 20,
        "SlowSpeed": 10,
        "LiftSpeed": 10,
        "LowerDownSpeed": 10
    }
    
    max_cycle = 1000
    actual_cycle = 1000

    for w_no, sequence in wagon_sequences.items():
        wagon = Wagon(w_no, 0, 30, 20, 10, 10, 10)
        processor = SequenceProcessor(wagon, max_cycle, actual_cycle)
        
        for instr, val_dist, stn_no in sequence:
            processor.process_instruction(instr, val_dist)
            
            # Since the requirement asks for STATION_NO in result, 
            # we need to track which station_no produced which dip_data entry.
            # SequenceProcessor uses the 'value' (distance) as key.
            
        # Merge results for this wagon into a global results table
        # We need to map distances back to station_nos
        wagon_results = processor.finalize_dip_times()
        for dist, data in wagon_results.items():
            # Find station_no for this distance
            s_no = None
            for s, d in stn_map.items():
                if d == dist:
                    s_no = s
                    break
            
            if s_no is not None:
                if s_no not in all_dip_data:
                    all_dip_data[s_no] = data
                else:
                    # If multiple wagons visit, we might need a more complex merge, 
                    # but for now, let's just take the last one or accumulate
                    # Requirement says "station_no: {DipIn, DipOut, DipTime}"
                    all_dip_data[s_no] = data

    # 4. Final Output Table
    table_data = []
    final_dict = {}
    for stn in sorted(all_dip_data.keys()):
        d = all_dip_data[stn]
        table_data.append([stn, d['DipInTime'], d['DipOutTime'], d['DipTime']])
        final_dict[stn] = d

    print(f"\nFinal Dip Time Calculation Table (Sequence: {seq_path}):")
    print(tabulate(table_data, headers=["Station No", "Dip In Time", "Dip Out Time", "Dip Time"], tablefmt="grid"))
    
    return final_dict

if __name__ == "__main__":
    process_sequence_csv("diptime/sequnce.csv", "tanks_csv_expanded.csv")
