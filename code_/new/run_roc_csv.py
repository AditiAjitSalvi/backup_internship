import csv
import time
from scheduling_ROC.domain.models import Row, Station, StationType, Transporter, Load, Process
from scheduling_ROC.services.plant_manager import PlantManager
from scheduling_ROC.api.interface import ROC_API

def load_plant_from_csv(api: ROC_API, station_csv: str):
    rows = {}
    stn_id_to_process = {}
    with open(station_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_no = row['Row No']
            if row_no not in rows:
                rows[row_no] = Row(row_id=f"Row{row_no}", transporter=Transporter(transporter_id=f"T{row_no}"))
            
            stn_id = row['Station No']
            p_name = row['Process Name']
            dist = float(row['Distance(mm)'])
            stn_id_to_process[stn_id] = p_name
            
            stn_type = StationType.TANK
            if "Loading" in p_name:
                stn_type = StationType.LOADING
            elif "Unloading" in p_name:
                stn_type = StationType.UNLOADING
            
            stn = Station(
                station_id=stn_id,
                station_type=stn_type,
                position_mm=dist,
                compatible_processes=[p_name]
            )
            rows[row_no].stations.append(stn)
    
    for r in rows.values():
        api.update_row_config(r)
    return stn_id_to_process

def schedule_from_sequence_csv(api: ROC_API, sequence_csv: str, stn_map: dict):
    wagon_processes = {}
    with open(sequence_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w_no = row['Wagon No']
            instr = row['Instruction']
            val = row['Instruction Value'].strip()
            
            if w_no not in wagon_processes:
                wagon_processes[w_no] = []
            
            if "Wait" in instr:
                if wagon_processes[w_no]:
                    wagon_processes[w_no][-1]["duration"] = float(val)
            elif "Put On" in instr:
                p_name = stn_map.get(val, f"Process_{val}")
                wagon_processes[w_no].append({"name": p_name, "duration": 0})

    for w_no, p_list in wagon_processes.items():
        print(f"Submitting load for Wagon {w_no} with {len(p_list)} steps.")
        api.submit_load(p_list)

if __name__ == "__main__":
    manager = PlantManager()
    api = ROC_API(manager)
    
    print("--- Loading Plant Configuration ---")
    stn_map = load_plant_from_csv(api, "test.csv")
    
    print("\n--- Scheduling Loads from Sequence ---")
    schedule_from_sequence_csv(api, "sequnce.csv", stn_map)
    
    print("\n--- Running Simulation ---")
    # Simulation loop
    for _ in range(30):
        time.sleep(1)
    
    print("\n--- Exporting Results ---")
    manager.export_logs("output.json")
    
    print("\nSimulation complete.")
