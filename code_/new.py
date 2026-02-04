#!/usr/bin/env python3

import os
import csv
import argparse
from collections import defaultdict
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
# CONFIGURATION & PATHS (UNCHANGED)
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

wagon_config_csv = os.path.join(parent_dir, "wagon_config.csv")
if not os.path.exists(wagon_config_csv):
    wagon_config_csv = r"e:\Internship\code\wagon_config.csv"

tanks_csv_default = os.path.join(base_dir, "tanks_csv.csv")
csv_programs = r"C:\Users\aditi\Downloads\Internship\code_\sequnce for traing.csv"
csv_zones = r"e:\Internship\WayTime-dB.mdb\CrossTrolleyMaster.csv"

MAX_STATIONS = 200
MAX_STEPS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD WAGON CONFIG (UNCHANGED)
# ==========================================
def load_wagon_config(config_path=wagon_config_csv):
    config = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row.get('Transporter Name') or row.get('Wagon Number')
            if not w:
                continue
            config[w.strip()] = {
                'min_stn': int(row.get('Minimum Station No', 1)),
                'max_stn': int(row.get('Maximum Station No', 200))
            }
    return config

# ============================================================
# 🔥 NEW FUNCTION – PLC SEQUENCE GENERATOR (ONLY ADDITION)
# ============================================================
def generate_plc_sequence(tanks, wagon_config):
    """
    Generates PLC-style wagon sequence exactly like required output
    """
    sequence = []
    step_counter = defaultdict(int)

    for i in range(len(tanks)):
        curr = tanks[i]
        curr_stn = int(curr['station_no'])

        # identify wagon using min/max station
        wagon = None
        for w, cfg in wagon_config.items():
            if cfg['min_stn'] <= curr_stn <= cfg['max_stn']:
                wagon = w
                break

        if not wagon:
            continue

        # GET FROM
        step_counter[wagon] += 1
        sequence.append([
            wagon,
            step_counter[wagon],
            "Get From",
            curr_stn
        ])

        # PUT ON (next station)
        if i + 1 < len(tanks):
            next_stn = int(tanks[i + 1]['station_no'])
            step_counter[wagon] += 1
            sequence.append([
                wagon,
                step_counter[wagon],
                "Put On",
                next_stn
            ])

        # WAIT FOR SEC
        dip = int(curr.get("dip_time_sec", 0))
        if dip > 0:
            step_counter[wagon] += 1
            sequence.append([
                wagon,
                step_counter[wagon],
                "Wait For Sec",
                dip
            ])

    return sequence

# ==========================================
# GENERATION MODE (MODIFIED PART ONLY)
# ==========================================
def generate_sequence_from_tanks(csv_path):
    print(f"\nProcessing tank file: {csv_path}")

    if not os.path.exists(csv_path):
        print("Tank file not found")
        return

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        tanks = list(reader)

    wagon_config = load_wagon_config()
    plc_sequence = generate_plc_sequence(tanks, wagon_config)

    print("\nGenerated PLC Sequence:\n")
    print(tabulate(
        plc_sequence,
        headers=["Wagon", "Step No", "Command", "Value"],
        tablefmt="grid"
    ))

# ==========================================
# MAIN (UNCHANGED LOGIC FLOW)
# ==========================================
def main():
    global wagon_config_csv

    parser = argparse.ArgumentParser(description="PLC Sequence Generator")
    parser.add_argument("--mode", choices=["gen"], default="gen")
    parser.add_argument("--input", default=tanks_csv_default)
    parser.add_argument("--config", default=wagon_config_csv)

    args = parser.parse_args()
    wagon_config_csv = args.config

    if args.mode == "gen":
        generate_sequence_from_tanks(args.input)

if __name__ == "__main__":
    main()
