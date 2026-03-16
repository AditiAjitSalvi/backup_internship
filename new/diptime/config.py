# Dip Time Calculation Parameters Configuration
# Edit these values to match your expected output

# =====================================================
# WAGON PARAMETERS
# =====================================================
# These should match your wagon configuration in the database

# Load from wagon_config.csv
import csv
import os


def load_wagon_config(config_file):
    """Load wagon configuration from CSV file."""
    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wagon_name = row.get("Transporter Name", "Wagon1").strip()
                config = {
                    "superfast_speed": float(row.get("Superfast Speed", 30)),
                    "fast_speed": float(row.get("Fast Speed", 20)),
                    "slow_speed": float(row.get("Slow Speed", 10)),
                    "lift_speed": float(row.get("Lift Time", 5)),
                    "lower_speed": float(row.get("Lower Time", 5)),
                    "row_number": int(row.get("Row Number", 1)),
                    "basic_position": int(row.get("Basic Position", 0)),
                }
                break  # Take first wagon
    return config


# Try to load from model folder
config_file = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model", "wagon_config.csv"
)
WAGON_CONFIG = load_wagon_config(config_file)

# Speed settings (mm/second)
SUPERFAST_SPEED = WAGON_CONFIG.get("superfast_speed", 30)
FAST_SPEED = WAGON_CONFIG.get("fast_speed", 20)
SLOW_SPEED = WAGON_CONFIG.get("slow_speed", 10)

# Lift and Lower down times (seconds)
LIFT_SPEED = WAGON_CONFIG.get("lift_speed", 5)
LOWER_SPEED = WAGON_CONFIG.get("lower_speed", 5)

# =====================================================
# SYSTEM PARAMETERS
# =====================================================

# Cross-trolley time (seconds) - time to move between rows
CT_TIME = 0

# Maximum cycle time in seconds (used for wrap-around calculation)
MAX_CYCLE_TIME = 600

# Actual operational cycle time (for scaling DipTimeActual)
ACTUAL_CYCLE_TIME = 600
