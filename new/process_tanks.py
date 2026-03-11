import csv
import json
from diptime_calculation import calculate_wagon_dip_times
from tabulate import tabulate

def process_tanks_csv(csv_path):
    # Load tanks data
    tanks = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tanks.append(row)
            
    # Define a default wagon
    wagon_params = {
        "BasicPosition": 0,
        "SuperfastSpeed": 30,
        "FastSpeed": 20,
        "SlowSpeed": 10,
        "LiftSpeed": 10,
        "LowerDownSpeed": 10
    }
    
    instructions = []
    
    # Map columns from test.csv
    # Station No, Process Name, Distance(mm)
    # test.csv doesn't have dip_time_sec, assuming 0 or using standard 60s for "Dip" names
    
    for i in range(len(tanks)):
        stn = tanks[i].get('Station No', tanks[i].get('station_no'))
        dist_str = tanks[i].get('Distance(mm)', tanks[i].get('distance_mm'))
        dist = float(dist_str) if dist_str else 0.0
        
        proc_name = tanks[i].get('Process Name', tanks[i].get('process_name', '')).upper()
        # Heuristic: if "DIP" is in name, add some dip time for testing, else use dip_time_sec if exists
        dip = float(tanks[i].get('dip_time_sec', 0))
        if "DIP" in proc_name and dip == 0:
            dip = 60.0 # Default 60s if name implies dip
            
        instructions.append(("PUT ON", dist))
        if dip > 0:
            instructions.append(("WAIT FOR SEC", dip))
        instructions.append(("GET FROM", dist))
        
    # Process
    results = calculate_wagon_dip_times(wagon_params, instructions, 1000, 1000)
    
    # Format as table
    table_data = []
    for i in range(len(tanks)):
        stn_no = tanks[i].get('Station No', tanks[i].get('station_no'))
        dist_str = tanks[i].get('Distance(mm)', tanks[i].get('distance_mm'))
        dist = float(dist_str) if dist_str else 0.0
        
        data = results.get(dist, {"DipInTime": 0, "DipOutTime": 0, "DipTime": 0})
        
        table_data.append([
            stn_no, 
            data['DipInTime'], 
            data['DipOutTime'], 
            data['DipTime']
        ])
    
    print(f"\nDip Time Calculation Table for {csv_path}:")
    print(tabulate(table_data, headers=["Station No", "Dip In Time", "Dip Out Time", "Dip Time"], tablefmt="grid"))
    
    print("\nDip Time Calculation Table:")
    print(tabulate(table_data, headers=["Station No", "Dip In Time", "Dip Out Time", "Dip Time"], tablefmt="grid"))

if __name__ == "__main__":
    process_tanks_csv("test.csv")
