from adv import generate_sequence_from_data

tanks = [
    {"station_no": "1", "process_name": "Loading", "dip_time_sec": "0"},
    {"station_no": "2", "process_name": "A acid", "dip_time_sec": "10"},
    {"station_no": "3", "process_name": "A acid", "dip_time_sec": "10"},
    {"station_no": "4", "process_name": "Rinse", "dip_time_sec": "5"}
]

config = {
    "W1": {
        "row": 1,
        "min_stn": 1,
        "max_stn": 200,
        "sf": 1000, "f": 500, "s": 100,
        "lift_time": 5, "lower_time": 5,
        "basic_pos": 0, "occupy_factor": 1.0
    }
}

try:
    generate_sequence_from_data(tanks, config=config, verbose=True)
    print("FINISHED")
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERROR:", e)
