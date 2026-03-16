import csv
import json
from tabulate import tabulate
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Import configuration
try:
    from config import (
        SUPERFAST_SPEED,
        FAST_SPEED,
        SLOW_SPEED,
        LIFT_SPEED,
        LOWER_SPEED,
        CT_TIME,
    )

    DEFAULT_PARAMS = {
        "SuperfastSpeed": SUPERFAST_SPEED,
        "FastSpeed": FAST_SPEED,
        "SlowSpeed": SLOW_SPEED,
        "LiftSpeed": LIFT_SPEED,
        "LowerDownSpeed": LOWER_SPEED,
    }
except ImportError:
    DEFAULT_PARAMS = {
        "SuperfastSpeed": 30,
        "FastSpeed": 20,
        "SlowSpeed": 10,
        "LiftSpeed": 5,
        "LowerDownSpeed": 5,
    }
    CT_TIME = 25

# Import the enhanced calculation engine
from diptime_calculation import (
    calculate_wagon_dip_times,
    calculate_multi_tank_dip_times,
    DistanceCalculator,
    CrossTrolley,
    SequenceProcessor,
)

# Enable debug mode
DEBUG_MODE = True


def load_station_data(station_csv: str) -> Dict:
    """Load station metadata from CSV file."""
    stations = {}
    try:
        with open(station_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle multiple possible column name variations
                stn_id = (
                    row.get("station_no")
                    or row.get("Station No")
                    or row.get("Station")
                    or row.get("Value")
                    or row.get("station")
                )
                if stn_id:
                    dist = (
                        row.get("distance_mm")
                        or row.get("Distance(mm)")
                        or row.get("Distance")
                        or row.get("distance")
                        or "0"
                    )
                    proc = (
                        row.get("process_name")
                        or row.get("Process Name")
                        or row.get("Process")
                        or row.get("process")
                        or ""
                    )
                    crit = (
                        row.get("critical_status")
                        or row.get("Criticality")
                        or row.get("Critical")
                        or row.get("critical")
                        or "Low"
                    )
                    flight_bar = (
                        row.get("FlightBar") or row.get("flight_bar") or "False"
                    )
                    row_no = (
                        row.get("Row") or row.get("row_no") or row.get("RowNo") or "1"
                    )
                    # Load expected dip time from CSV
                    dip_time_sec = (
                        row.get("dip_time_sec") or row.get("DipTimeSec") or "0"
                    )

                    stations[str(stn_id)] = {
                        "Process Name": proc,
                        "Criticality": crit,
                        "Distance": dist,
                        "FlightBar": flight_bar.lower() == "true",
                        "RowNo": int(row_no) if str(row_no).isdigit() else 1,
                        "DipTimeSec": int(dip_time_sec)
                        if str(dip_time_sec).isdigit()
                        else 0,
                    }
    except Exception as e:
        print(f"Error reading station file: {e}")
    return stations


def load_sequence_data(sequence_csv: str) -> Tuple[List[Dict], Dict[str, List[Tuple]]]:
    """
    Load sequence data from CSV.
    Returns: (raw_sequence_list, wagon_sequences_dict)
    """
    raw_sequence = []
    wagon_sequences = defaultdict(list)
    running_time = 0.0

    try:
        with open(sequence_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle multiple possible column name variations
                wagon = (
                    row.get("Wagon")
                    or row.get("Wagon No")
                    or row.get("WagonNumber")
                    or row.get("wagon")
                    or "Wagon1"
                )
                instr = (
                    (
                        row.get("Instrution")
                        or row.get("Instruction")
                        or row.get("Command")
                        or row.get("Action")
                        or row.get("instruction")
                        or ""
                    )
                    .upper()
                    .strip()
                )
                val = str(
                    row.get("Value")
                    or row.get("Instruction Value")
                    or row.get("Station")
                    or row.get("instruction_value")
                    or ""
                ).strip()

                # Try to get time, calculate if not present
                time_key = next(
                    (
                        k
                        for k in row.keys()
                        if k and ("time" in k.lower() or "acc" in k.lower())
                    ),
                    None,
                )
                try:
                    if time_key:
                        running_time = float(row.get(time_key, 0))
                    else:
                        # Calculate estimated time based on instructions
                        running_time = estimate_instruction_time(
                            instr, val, running_time
                        )
                except:
                    running_time = estimate_instruction_time(instr, val, running_time)

                raw_entry = {
                    "wagon": wagon,
                    "instruction": instr,
                    "value": val,
                    "time": running_time,
                }
                raw_sequence.append(raw_entry)

                # Add to wagon-specific sequence
                if val:
                    try:
                        int_val = int(float(val))
                    except:
                        int_val = 0
                    wagon_sequences[wagon].append((instr, int_val))

    except Exception as e:
        print(f"Error reading sequence file: {e}")

    return raw_sequence, wagon_sequences


def estimate_instruction_time(instr: str, value: str, current_time: float) -> float:
    """Estimate time for an instruction if no time data is provided."""
    cmd = instr.upper()

    # Default times for various instructions
    if "GET" in cmd or "PUT" in cmd:
        return current_time + 15  # ~15 sec for get/put movement
    elif "WAIT" in cmd or "SEC" in cmd:
        try:
            return current_time + float(value)
        except:
            return current_time + 10
    elif "CLAMP" in cmd or "DECLAMP" in cmd:
        return current_time + 5
    elif "TILT" in cmd or "UNTILT" in cmd:
        return current_time + 10
    elif "SET INT" in cmd or "WAIT INT" in cmd:
        return current_time + 2
    elif "SET CT" in cmd:
        return current_time + 20  # Cross trolley time
    else:
        return current_time + 5  # Default


def load_wagon_data(
    wagon_csv: str = None, default_wagon: Dict = None
) -> Dict[str, Dict]:
    """Load wagon parameters from CSV or use defaults."""
    wagons = {}

    if wagon_csv:
        try:
            with open(wagon_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    wagon_id = row.get("Wagon No") or row.get("Wagon") or "Wagon1"
                    wagons[wagon_id] = {
                        "WagonId": wagon_id,
                        "BasicPosition": int(row.get("BasicPosition", 0)),
                        "RowNo": int(row.get("RowNo", 1)),
                        "SuperfastSpeed": float(row.get("SuperfastSpeed", 30)),
                        "FastSpeed": float(row.get("FastSpeed", 20)),
                        "SlowSpeed": float(row.get("SlowSpeed", 10)),
                        "LiftSpeed": float(row.get("LiftSpeed", 5)),
                        "LowerDownSpeed": float(row.get("LowerDownSpeed", 5)),
                    }
        except Exception as e:
            print(f"Error reading wagon file: {e}")

    if not wagons and default_wagon:
        wagons["Wagon1"] = default_wagon

    return wagons


def load_distance_data(distance_csv: str = None) -> Dict:
    """Load distance data between positions for zone calculations."""
    distance_data = {}

    if distance_csv:
        try:
            with open(distance_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    from_pos = int(row.get("FromPosition", 0))
                    to_pos = int(row.get("ToPosition", 0))
                    row_no = int(row.get("RowNo", 1))
                    dist1 = float(row.get("Distance1", 0))
                    dist2 = float(row.get("Distance2", 0))
                    sensor_dist = float(row.get("SensorDistance", 50))
                    flightbar = row.get("FlightBar", "False").lower() == "true"

                    key = (from_pos, to_pos, row_no)
                    distance_data[key] = {
                        "distance1": dist1,
                        "distance2": dist2,
                        "censor_distance": sensor_dist,
                        "flightbar": flightbar,
                    }
        except Exception as e:
            print(f"Error reading distance file: {e}")

    return distance_data


def load_cross_trolley_data(ct_csv: str = None) -> Dict:
    """Load cross-trolley configuration data."""
    ct_data = {}

    if ct_csv:
        try:
            with open(ct_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ct_station = int(row.get("CTStation", 0))
                    row1_station = int(row.get("Row1StationNo", 0))
                    row2_station = int(row.get("Row2StationNo", 0))
                    speed = float(row.get("Speed", 10))

                    ct_data[ct_station] = {
                        "row1_station_no": row1_station,
                        "row2_station_no": row2_station,
                        "speed": speed,
                    }
        except Exception as e:
            print(f"Error reading cross-trolley file: {e}")

    return ct_data


def generate_dip_time_table(
    station_csv: str,
    sequence_csv: str,
    max_cycle_time: float = 600,
    actual_cycle_time: float = None,
    wagon_csv: str = None,
    distance_csv: str = None,
    ct_csv: str = None,
) -> List[Dict]:
    """
    Generates a Dip Time Table using the enhanced calculation engine.

    This function combines station data, sequence instructions, wagon parameters,
    and distance data to calculate accurate dip times.

    Args:
        station_csv: Path to station metadata CSV
        sequence_csv: Path to sequence instructions CSV
        max_cycle_time: Design cycle time for the system
        actual_cycle_time: Actual operational cycle time (defaults to max_cycle_time)
        wagon_csv: Optional path to wagon parameters CSV
        distance_csv: Optional path to distance data CSV
        ct_csv: Optional path to cross-trolley data CSV

    Returns:
        List of dictionaries containing dip time results
    """
    if actual_cycle_time is None:
        actual_cycle_time = max_cycle_time

    # Load all data
    stations = load_station_data(station_csv)
    raw_sequence, wagon_sequences = load_sequence_data(sequence_csv)

    # Default wagon parameters using config values
    default_wagon = {
        "WagonId": "Wagon1",
        "BasicPosition": 0,
        "RowNo": 1,
        "SuperfastSpeed": SUPERFAST_SPEED,
        "FastSpeed": FAST_SPEED,
        "SlowSpeed": SLOW_SPEED,
        "LiftSpeed": LIFT_SPEED,
        "LowerDownSpeed": LOWER_SPEED,
    }

    # Get row info and positions from stations
    station_rows = {}
    station_positions = {}
    station_dip_times = {}
    for stn_id, stn_data in stations.items():
        try:
            station_rows[int(stn_id)] = stn_data.get("RowNo", 1)
            # Convert distance string to int (in mm)
            dist_str = stn_data.get("Distance", "0")
            station_positions[int(stn_id)] = (
                int(dist_str) if dist_str.isdigit() else int(stn_id)
            )
            # Get expected dip time from CSV
            station_dip_times[int(stn_id)] = stn_data.get("DipTimeSec", 0)
        except:
            pass

    # Determine row for each wagon based on stations visited
    wagon_rows = {}
    for wagon_id, seq in wagon_sequences.items():
        row_nos = []
        for instr, val in seq:
            if val in station_rows:
                row_nos.append(station_rows[val])
        # Use most common row or default to 1
        if row_nos:
            wagon_rows[wagon_id] = max(set(row_nos), key=row_nos.count)
        else:
            wagon_rows[wagon_id] = 1

    wagons = load_wagon_data(wagon_csv, default_wagon)

    # Normalize wagon IDs (remove extra spaces)
    normalized_sequences = {}
    for wagon_id, seq in wagon_sequences.items():
        # Normalize wagon ID
        normalized_id = wagon_id.strip()
        normalized_sequences[normalized_id] = seq

    # Update wagon row numbers from station data
    for wagon_id in wagons:
        if wagon_id in wagon_rows:
            wagons[wagon_id]["RowNo"] = wagon_rows[wagon_id]
        # Also try with/without space
        if " " in wagon_id:
            alt_id = wagon_id.replace(" ", "")
        else:
            alt_id = "Wagon " + wagon_id if wagon_id.isdigit() else wagon_id
        if alt_id in normalized_sequences and wagon_id in wagons:
            # Get row from stations visited
            row_nos = []
            for instr, val in normalized_sequences[alt_id]:
                if str(val) in station_rows:
                    row_nos.append(station_rows[str(val)].get("RowNo", 1))
            if row_nos:
                wagons[wagon_id]["RowNo"] = max(set(row_nos), key=row_nos.count)

    distance_data = load_distance_data(distance_csv)
    ct_data = load_cross_trolley_data(ct_csv)

    # Calculate dip times - normalize keys
    if len(wagons) == 1:
        # Single wagon mode - try to find the wagon in sequences
        wagon_id = list(wagons.keys())[0]

        # Try different variations of wagon ID
        seq_key = wagon_id
        if wagon_id not in normalized_sequences:
            # Try with/without space
            if " " in wagon_id:
                seq_key = wagon_id.replace(" ", "")
            else:
                seq_key = "Wagon " + wagon_id

        if seq_key in normalized_sequences:
            wagon_params = wagons[wagon_id]
            instructions = normalized_sequences[seq_key]

            results = calculate_wagon_dip_times(
                wagon_params,
                instructions,
                max_cycle_time,
                actual_cycle_time,
                distance_data,
                ct_data,
                station_rows,
                station_positions,
                station_dip_times,
            )
        else:
            # Try all sequences
            for k, v in normalized_sequences.items():
                if v:
                    wagon_params = wagons.get(
                        k, wagons.get(list(wagons.keys())[0], default_wagon)
                    )
                    results = calculate_wagon_dip_times(
                        wagon_params,
                        v,
                        max_cycle_time,
                        actual_cycle_time,
                        distance_data,
                        ct_data,
                        station_rows,
                        station_positions,
                        station_dip_times,
                    )
                    break
            else:
                results = {}
    else:
        # Multi-wagon/multi-tank mode
        sequences = {}
        for wid in wagons.keys():
            seq_key = wid
            if wid not in normalized_sequences:
                if " " in wid:
                    seq_key = wid.replace(" ", "")
                else:
                    seq_key = "Wagon " + wid if wid.isdigit() else wid
            sequences[wid] = normalized_sequences.get(seq_key, [])

            results = calculate_multi_tank_dip_times(
                wagons,
                sequences,
                max_cycle_time,
                actual_cycle_time,
                distance_data,
                ct_data,
            )

    # Format results with station metadata
    final_table = []

    def sort_key(item):
        key = item[0]
        if isinstance(key, int):
            return key
        if isinstance(key, str) and key.isdigit():
            return int(key)
        return key

    for station_no, times in sorted(results.items(), key=sort_key):
        stn_meta = stations.get(str(station_no), {})

        # Determine wagon from the result
        wagon_id = times.get("Wagon", "Wagon1")

        final_table.append(
            {
                "Wagon": wagon_id,
                "Station": station_no,
                "Process Name": stn_meta.get("Process Name", "Unknown"),
                "Criticality": stn_meta.get("Criticality", "Low"),
                "DipIn": times["DipInTime"],
                "DipOut": times["DipOutTime"],
                "DipTime": times["DipTime"],
                "DipTimeActual": times["DipTimeActual"],
                "FlightBar": times.get("FlightBar", False),
                "Distance": stn_meta.get("Distance", "0"),
            }
        )

    # Sort by wagon then station
    def table_sort_key(x):
        wagon_num = 0
        try:
            wagon_str = x["Wagon"]
            if " " in wagon_str:
                wagon_num = int(wagon_str.split(" ")[-1])
            elif wagon_str[5:].isdigit():
                wagon_num = int(wagon_str[5:])
        except:
            pass

        station_num = 0
        try:
            station_num = int(x["Station"])
        except:
            pass

        return (wagon_num, station_num)

    final_table.sort(key=table_sort_key)

    return final_table


def generate_dip_time_from_csv(
    station_csv: str,
    sequence_csv: str,
    max_cycle_time: float = 600,
    actual_cycle_time: float = None,
    show_table: bool = True,
) -> List[Dict]:
    """
    Convenience function to generate dip time table with default parameters.
    """
    return generate_dip_time_table(
        station_csv, sequence_csv, max_cycle_time, actual_cycle_time
    )


def display_results(data: List[Dict], show_actual: bool = True):
    """Prints the final table using tabulate."""
    if not data:
        print("No data to display.")
        return

    if show_actual:
        headers = [
            "Wagon",
            "Station",
            "Process Name",
            "Criticality",
            "DipIn",
            "DipOut",
            "DipTime",
            "DipTimeActual",
            "FlightBar",
        ]
        rows = []
        for d in data:
            rows.append(
                [
                    d["Wagon"],
                    d["Station"],
                    d["Process Name"],
                    d["Criticality"],
                    d["DipIn"],
                    d["DipOut"],
                    d["DipTime"],
                    d["DipTimeActual"],
                    "Yes" if d.get("FlightBar", False) else "No",
                ]
            )
    else:
        headers = [
            "Wagon",
            "Station",
            "Process Name",
            "Criticality",
            "DipIn",
            "DipOut",
            "DipTime",
        ]
        rows = []
        for d in data:
            rows.append(
                [
                    d["Wagon"],
                    d["Station"],
                    d["Process Name"],
                    d["Criticality"],
                    d["DipIn"],
                    d["DipOut"],
                    d["DipTime"],
                ]
            )

    print("\n" + "=" * 80)
    print("INDUSTRIAL DIP TIME CALCULATION TABLE")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def export_to_csv(data: List[Dict], output_file: str):
    """Export results to CSV file."""
    if not data:
        print("No data to export.")
        return

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        print(f"Results exported to: {output_file}")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def export_to_json(data: List[Dict], output_file: str):
    """Export results to JSON file."""
    if not data:
        print("No data to export.")
        return

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Results exported to: {output_file}")
    except Exception as e:
        print(f"Error exporting to JSON: {e}")


if __name__ == "__main__":
    import os

    # Determine paths
    base_dir = os.path.dirname(__file__)
    stations_path = os.path.join(base_dir, "test.csv")
    sequence_path = os.path.join(base_dir, "sequnce.csv")

    # Check for files
    if not os.path.exists(stations_path):
        print(f"Station file not found: {stations_path}")
        print("Creating sample station file...")

        # Create sample station data
        sample_stations = [
            {
                "Station No": "1",
                "Process Name": "Loading",
                "Criticality": "High",
                "Distance(mm)": "0",
                "FlightBar": "False",
                "RowNo": "1",
            },
            {
                "Station No": "100",
                "Process Name": "Pre-Clean",
                "Criticality": "Medium",
                "Distance(mm)": "5000",
                "FlightBar": "True",
                "RowNo": "1",
            },
            {
                "Station No": "500",
                "Process Name": "Cleaning",
                "Criticality": "High",
                "Distance(mm)": "15000",
                "FlightBar": "True",
                "RowNo": "1",
            },
            {
                "Station No": "1000",
                "Process Name": "Main Tank",
                "Criticality": "Critical",
                "Distance(mm)": "25000",
                "FlightBar": "True",
                "RowNo": "1",
            },
            {
                "Station No": "1500",
                "Process Name": "Rinse",
                "Criticality": "Medium",
                "Distance(mm)": "35000",
                "FlightBar": "True",
                "RowNo": "1",
            },
            {
                "Station No": "2000",
                "Process Name": "Drying",
                "Criticality": "Low",
                "Distance(mm)": "45000",
                "FlightBar": "False",
                "RowNo": "1",
            },
            {
                "Station No": "2500",
                "Process Name": "Unloading",
                "Criticality": "High",
                "Distance(mm)": "50000",
                "FlightBar": "False",
                "RowNo": "1",
            },
        ]

        with open(stations_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sample_stations[0].keys())
            writer.writeheader()
            writer.writerows(sample_stations)
        print(f"Created: {stations_path}")

    if not os.path.exists(sequence_path):
        print(f"Sequence file not found: {sequence_path}")
        print("Creating sample sequence file...")

        # Create sample sequence
        sample_sequence = [
            {
                "Wagon No": "1",
                "Instruction": "GET FROM",
                "Instruction Value": "1",
                "Time": "0",
            },
            {
                "Wagon No": "1",
                "Instruction": "PUT ON",
                "Instruction Value": "100",
                "Time": "15",
            },
            {
                "Wagon No": "1",
                "Instruction": "WAIT FOR SEC",
                "Instruction Value": "30",
                "Time": "45",
            },
            {
                "Wagon No": "1",
                "Instruction": "GET FROM",
                "Instruction Value": "100",
                "Time": "75",
            },
            {
                "Wagon No": "1",
                "Instruction": "PUT ON",
                "Instruction Value": "500",
                "Time": "105",
            },
            {
                "Wagon No": "1",
                "Instruction": "WAIT FOR SEC",
                "Instruction Value": "45",
                "Time": "150",
            },
            {
                "Wagon No": "1",
                "Instruction": "GET FROM",
                "Instruction Value": "500",
                "Time": "195",
            },
            {
                "Wagon No": "1",
                "Instruction": "PUT ON",
                "Instruction Value": "1000",
                "Time": "225",
            },
            {
                "Wagon No": "1",
                "Instruction": "WAIT FOR SEC",
                "Instruction Value": "60",
                "Time": "285",
            },
            {
                "Wagon No": "1",
                "Instruction": "GET FROM",
                "Instruction Value": "1000",
                "Time": "345",
            },
            {
                "Wagon No": "1",
                "Instruction": "PUT ON",
                "Instruction Value": "1500",
                "Time": "375",
            },
            {
                "Wagon No": "1",
                "Instruction": "WAIT FOR SEC",
                "Instruction Value": "30",
                "Time": "405",
            },
            {
                "Wagon No": "1",
                "Instruction": "GET FROM",
                "Instruction Value": "1500",
                "Time": "435",
            },
            {
                "Wagon No": "1",
                "Instruction": "PUT ON",
                "Instruction Value": "2000",
                "Time": "465",
            },
            {
                "Wagon No": "1",
                "Instruction": "WAIT FOR SEC",
                "Instruction Value": "25",
                "Time": "490",
            },
            {
                "Wagon No": "1",
                "Instruction": "GET FROM",
                "Instruction Value": "2000",
                "Time": "515",
            },
            {
                "Wagon No": "1",
                "Instruction": "PUT ON",
                "Instruction Value": "2500",
                "Time": "545",
            },
        ]

        with open(sequence_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sample_sequence[0].keys())
            writer.writeheader()
            writer.writerows(sample_sequence)
        print(f"Created: {sequence_path}")

    # Run calculation with max cycle time of 600 seconds
    print("\nCalculating dip times...")
    print(f"Station file: {stations_path}")
    print(f"Sequence file: {sequence_path}")
    print(f"Max cycle time: 600 seconds")

    results = generate_dip_time_table(stations_path, sequence_path, max_cycle_time=600)
    display_results(results)

    # Export options
    print("\nExport options:")
    print("  - To CSV: Call export_to_csv(results, 'output.csv')")
    print("  - To JSON: Call export_to_json(results, 'output.json')")
