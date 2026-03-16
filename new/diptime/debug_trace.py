"""
Debug script to trace step-by-step calculation
"""

import csv
from diptime_calculation import (
    Wagon,
    DistanceCalculator,
    SequenceProcessor,
    InterlockManager,
    CrossTrolley,
)

# Load station data
stations = {}
with open("test.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        stn_id = int(row["station_no"])
        # Handle different column names
        row_num = row.get("Row") or row.get("row") or "1"
        stations[stn_id] = {
            "distance": int(row["distance_mm"]),
            "row": int(row_num),
            "dip_time": int(row["dip_time_sec"]),
        }

# Wagon config from model
wagon_params = {
    "WagonId": "Wagon1",
    "BasicPosition": 1,
    "RowNo": 1,
    "SuperfastSpeed": 12,
    "FastSpeed": 25,
    "SlowSpeed": 5,
    "LiftSpeed": 12,
    "LowerDownSpeed": 12,
}

# Station positions
station_positions = {k: v["distance"] for k, v in stations.items()}
station_rows = {k: v["row"] for k, v in stations.items()}
station_dip_times = {k: v["dip_time"] for k, v in stations.items()}

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Wagon: {wagon_params}")
print(f"Station Positions: {station_positions}")
print(f"Station Rows: {station_rows}")
print(f"Station Dip Times: {station_dip_times}")

# Create processor with debug
wagon = Wagon(
    wagon_id=wagon_params["WagonId"],
    basic_pos=wagon_params["BasicPosition"],
    row_no=wagon_params["RowNo"],
    sf_speed=wagon_params["SuperfastSpeed"],
    f_speed=wagon_params["FastSpeed"],
    s_speed=wagon_params["SlowSpeed"],
    lift_speed=wagon_params["LiftSpeed"],
    lower_speed=wagon_params["LowerDownSpeed"],
)

distance_calc = DistanceCalculator({}, station_positions)
processor = SequenceProcessor(
    wagon,
    600,
    600,
    distance_calc,
    InterlockManager(),
    CrossTrolley({}),
    station_rows,
    station_dip_times,
    debug=True,
)

# Sequence from CSV
instructions = [
    ("GET FROM", 1),
    ("PUT ON", 2),
    ("GET FROM", 1),
    ("PUT ON", 3),
    ("GET FROM", 1),
    ("PUT ON", 4),
    ("GET FROM", 2),
    ("PUT ON", 5),
    ("GET FROM", 5),
    ("PUT ON", 6),
    ("GET FROM", 3),
    ("PUT ON", 5),
    ("GET FROM", 6),
    ("PUT ON", 7),
    ("GET FROM", 5),
    ("PUT ON", 6),
    ("GET FROM", 7),
    ("PUT ON", 8),
]

print("\n" + "=" * 60)
print("SEQUENCE EXECUTION")
print("=" * 60)

for cmd, val in instructions:
    processor.process_instruction(cmd, val)

# Print debug log
processor.print_debug_log()

# Final results
results = processor.finalize_dip_times()

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"{'Station':<10} {'DipIn':<10} {'DipOut':<10} {'DipTime':<10}")
print("-" * 40)
for stn in sorted(results.keys()):
    r = results[stn]
    print(f"{stn:<10} {r['DipInTime']:<10} {r['DipOutTime']:<10} {r['DipTime']:<10}")

print("\n" + "=" * 60)
print("EXPECTED RESULTS")
print("=" * 60)
print("Station 1: DipIn=0, DipOut=94")
print("Station 2: DipIn=29, DipOut=144, DipTime=115")
print("Station 3: DipIn=71, DipOut=220, DipTime=149")
print("Station 4: DipIn=124, DipOut=435, DipTime=311")
print("Station 5: DipIn=450, DipOut=520, DipTime=70")
print("Station 6: DipIn=535, DipOut=547, DipTime=12")
print("Station 7: DipIn=569, DipOut=581, DipTime=12")
print("Station 8: DipIn=595, DipOut=0")
