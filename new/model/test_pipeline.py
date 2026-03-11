import sys
from adv import gap_analysis_sequence

def test_pipeline():
    print("Running Pipeline Test...")
    
    # Minimal config for 1 wagon
    config = {
        "W1": {
            "row": "1",
            "min_stn": 1,
            "max_stn": 10,
            "sf": 30, "f": 20, "s": 10,
            "lift_time": 5, "lower_time": 5,
            "basic_pos": 1
        }
    }
    
    # Tanks with a parallel group at stations 2 and 3
    tanks = [
        {"station_no": "1", "Process Name": "Loading", "distance_mm": "0", "dip_time_sec": "0", "Row": "1"},
        {"station_no": "2", "Process Name": "Acid A", "distance_mm": "1000", "dip_time_sec": "30", "Row": "1"},
        {"station_no": "3", "Process Name": "Acid A", "distance_mm": "2000", "dip_time_sec": "30", "Row": "1"},
        {"station_no": "4", "Process Name": "Rinse", "distance_mm": "3000", "dip_time_sec": "10", "Row": "1"}
    ]
    
    seq = gap_analysis_sequence(tanks, config, enable_collision_engine=True, safe_distance_mm=500)
    
    print("\nGenerated Sequence:")
    for row in seq:
        if isinstance(row, list) and len(row) > 3:
            cmd = row[3]
            stn = row[4]
            acc = row[6]
            print(f"  Step {row[2]:>2} | {cmd:<20} | Stn: {stn} | Acc: {acc}")
            
test_pipeline()
