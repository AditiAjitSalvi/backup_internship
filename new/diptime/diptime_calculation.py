import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


# Global station configuration
station_positions = {}  # station_no -> position_mm
station_rows = {}  # station_no -> row_no
sensor_distances = {}  # station_no -> sensor_distance


def load_station_config(tanks: List[Dict]) -> None:
    """
    Load station positions and configuration from tanks data.
    Mimics database StationMaster loading from C#.
    """
    global station_positions, station_rows, sensor_distances

    station_positions = {}
    station_rows = {}
    sensor_distances = {}

    for tank in tanks:
        stn = int(tank.get("station_no", 0))
        if stn == 0:
            continue
        station_positions[stn] = float(tank.get("distance_mm", stn * 1000))
        station_rows[stn] = int(tank.get("Row") or tank.get("row") or 1)
        sensor_distances[stn] = int(tank.get("sensor_distance") or 50)


def get_distance_csharp(current_pos: int, dest_pos: int, row_no: int = 1) -> Dict:
    """
    C#-style distance calculation using 2-stage approach.
    Mimics: getDistance(currentposition, destinationValue ± 1) from frmDipTime.cs

    The C# code calls getDistance twice:
    1. getDistance(currentposition, destinationValue ± 1) -> d1, d2, censor_distance
    2. getDistance(destinationValue ± 1, destinationValue) -> d3, d4

    Final zones:
    - distance1 (superfast) = |d1 - d2|
    - distance2 (fast) = |d3 - d4| - censor_distance
    - distance3 (slow/sensor) = censor_distance
    """
    global station_positions, sensor_distances

    if current_pos == dest_pos:
        return {
            "distance1": 0,
            "distance2": 0,
            "censor_distance": 50,
            "flightbar": False,
        }

    # Determine direction (same as C#)
    step = 1 if dest_pos > current_pos else -1

    # Get positions
    pos_current = station_positions.get(current_pos, current_pos * 1000)
    intermediate = dest_pos + step
    pos_intermediate = station_positions.get(intermediate, intermediate * 1000)
    pos_dest = station_positions.get(dest_pos, dest_pos * 1000)

    # Stage 1: getDistance(currentposition, destinationValue ± 1)
    d1 = abs(pos_intermediate - pos_current)  # Distance to intermediate
    d2 = 0  # At intermediate point
    censor_dist = sensor_distances.get(dest_pos, 50)

    # Stage 2: getDistance(destinationValue ± 1, destinationValue)
    d3 = abs(pos_dest - pos_intermediate)  # Approach distance
    d4 = 0  # At exact destination

    # Calculate final zones (matching C# logic from frmDipTime.cs lines 400-421)
    if d1 > d2:
        distance1 = d1 - d2
    else:
        distance1 = d2 - d1

    if d3 > d4:
        distance2 = d3 - d4 - censor_dist
    else:
        distance2 = d4 - d3 - censor_dist

    distance2 = max(0, distance2)  # Ensure non-negative
    distance3 = censor_dist  # Slow/sensor zone

    return {
        "distance1": distance1,
        "distance2": distance2,
        "censor_distance": distance3,
        "flightbar": False,
    }


def calculate_time_value(
    distance1: float,
    distance2: float,
    distance3: float,
    sfspeed: float,
    fspeed: float,
    sspeed: float,
) -> float:
    """
    Calculate travel time using C# formula:
    time = (distance1 / (sfspeed * 16.66)) +
           (distance2 / (fspeed * 16.66)) +
           (distance3 / (sspeed * 16.66))
    """
    try:
        sfs = sfspeed * 16.66 if sfspeed > 0 else 1.0
        fs = fspeed * 16.66 if fspeed > 0 else 1.0
        ss = sspeed * 16.66 if sspeed > 0 else 1.0
        return (distance1 / sfs) + (distance2 / fs) + (distance3 / ss)
    except:
        return 0


def calculate_time_csharp(
    current_pos: int,
    dest_pos: int,
    row_no: int,
    sfspeed: float,
    fspeed: float,
    sspeed: float,
) -> float:
    """
    C#-style travel time calculation using 2-stage distance approach.
    Mimics DipTimeCalculation() from frmDipTime.cs
    """
    if current_pos == dest_pos:
        return 0.0

    dist_data = get_distance_csharp(current_pos, dest_pos, row_no)

    return calculate_time_value(
        dist_data["distance1"],
        dist_data["distance2"],
        dist_data["censor_distance"],
        sfspeed,
        fspeed,
        sspeed,
    )


class Wagon:
    """Represents an industrial wagon with its speed parameters and position."""

    def __init__(
        self,
        wagon_id: str,
        basic_pos: int,
        row_no: int,
        sf_speed: float,
        f_speed: float,
        s_speed: float,
        lift_speed: float,
        lower_speed: float,
    ):
        self.wagon_id = wagon_id
        self.basic_position = basic_pos
        self.row_no = row_no
        self.superfast_speed = sf_speed
        self.fast_speed = f_speed
        self.slow_speed = s_speed
        self.lift_speed = lift_speed
        self.lower_down_speed = lower_speed
        self.current_position = basic_pos


class DistanceCalculator:
    """Handles distance calculations between positions using zone-based approach."""

    def __init__(self, distance_data: Dict, station_positions: Dict[int, int] = None):
        """
        distance_data: Dict with key (from_pos, to_pos, row_no) -> {distance, flightbar, censor_distance}
        station_positions: Dict station_no -> position (distance in mm)
        """
        self.distance_data = distance_data
        self.station_positions = station_positions or {}  # station_no -> position

    def get_position(self, station_no: int) -> int:
        """Get the position (in mm) for a station number."""
        return self.station_positions.get(
            station_no, station_no
        )  # fallback to station_no as position

    def get_distance(self, from_pos: int, to_pos: int, row_no: int) -> Dict:
        """
        Get distance data between two positions.
        Returns dict with distance1, distance2, distance3 (slow zone), flightbar, censor_distance
        """
        key = (from_pos, to_pos, row_no)
        if key in self.distance_data:
            return self.distance_data[key]

        # Try reverse direction
        key_rev = (to_pos, from_pos, row_no)
        if key_rev in self.distance_data:
            return self.distance_data[key_rev]

        # Default fallback
        return {
            "distance1": abs(to_pos - from_pos),
            "distance2": 0,
            "distance3": 50,  # Default sensor distance
            "flightbar": False,
            "censor_distance": 50,
        }

    def calculate_travel_time(
        self, from_pos: int, to_pos: int, row_no: int, wagon: Wagon
    ) -> Tuple[float, bool, int]:
        """
        Calculate travel time using zone-based formula.

        Zone distribution:
        - Slow zone: last 50 units (sensor distance)
        - Fast zone: middle 1000 units
        - Superfast zone: first remaining units
        """
        from_position = self.get_position(from_pos)
        to_position = self.get_position(to_pos)

        if from_position == to_position:
            return (0.0, False, 0)

        distance_total = abs(to_position - from_position)

        # Zone distribution - 50 for sensor
        d3 = min(distance_total, 50.0)  # Slow zone (sensor)
        remaining = distance_total - d3

        d2 = min(remaining, 1000.0)  # Fast zone max 1000
        d1 = remaining - d2  # Superfast zone

        # Calculate time
        t1 = d1 / (wagon.superfast_speed * 16.66) if wagon.superfast_speed > 0 else 0
        t2 = d2 / (wagon.fast_speed * 16.66) if wagon.fast_speed > 0 else 0
        t3 = d3 / (wagon.slow_speed * 16.66) if wagon.slow_speed > 0 else 0

        travel_time = t1 + t2 + t3
        return (travel_time, False, int(d3))

    def calculate_travel_time_csharp(
        self, from_pos: int, to_pos: int, row_no: int, wagon: Wagon
    ) -> Tuple[float, bool, int]:
        """
        C#-style travel time calculation using 2-stage distance approach.
        Mimics DipTimeCalculation() from frmDipTime.cs
        """
        # Use global C#-style distance calculation
        global station_positions, sensor_distances

        if from_pos == to_pos:
            return (0.0, False, 0)

        # Use the C#-style distance calculation
        dist_data = get_distance_csharp(from_pos, to_pos, row_no)

        # Calculate time using C# formula
        travel_time = calculate_time_value(
            dist_data["distance1"],
            dist_data["distance2"],
            dist_data["censor_distance"],
            wagon.superfast_speed,
            wagon.fast_speed,
            wagon.slow_speed,
        )

        return (travel_time, dist_data["flightbar"], dist_data["censor_distance"])


class InterlockManager:
    """Manages interlock logic between wagons."""

    def __init__(self):
        self.interlock_times = {}  # interlock_no -> time_when_set

    def set_interlock(self, interlock_no: int, time_value: float):
        """Set an interlock at a specific time."""
        self.interlock_times[interlock_no] = time_value

    def get_interlock_time(self, interlock_no: int) -> Optional[float]:
        """Get the time when an interlock was set."""
        return self.interlock_times.get(interlock_no)


class CrossTrolley:
    """Handles cross-trolley movement calculations."""

    def __init__(self, ct_data: Dict[int, Dict]):
        """
        ct_data: ct_station_no -> {row1_station_no, row2_station_no, speed}
        """
        self.ct_data = ct_data

    def calculate_ct_time(self, ct_station_no: int) -> float:
        """Calculate cross-trolley travel time."""
        if ct_station_no not in self.ct_data:
            return 0.0

        ct_info = self.ct_data[ct_station_no]
        speed = ct_info.get("speed", 0)

        if speed > 0:
            # Standard cross-trolley distance is 2230mm
            return 2230 / (speed * 16.66)
        return 0.0


class SequenceProcessor:
    """Processes a sequence of instructions to calculate station-wise dip times."""

    def __init__(
        self,
        wagon: Wagon,
        max_cycle_time: float,
        actual_cycle_time: float,
        distance_calc: DistanceCalculator = None,
        interlock_manager: InterlockManager = None,
        cross_trolley: CrossTrolley = None,
        station_rows: Dict[int, int] = None,
        station_dip_times: Dict[int, int] = None,
        debug: bool = False,
    ):
        self.wagon = wagon
        self.max_cycle_time = max_cycle_time
        self.actual_cycle_time = actual_cycle_time
        self.total_time = 0.0
        self.dip_data = {}  # station_no -> {FirstDipInTime, LastDipOutTime, DipTime, FlightBar}

        # Use provided calculators or defaults
        self.distance_calc = distance_calc or DistanceCalculator({})
        self.interlock_manager = interlock_manager or InterlockManager()
        self.cross_trolley = cross_trolley or CrossTrolley({})

        # Station row mapping for CT detection
        self.station_rows = station_rows or {}

        # Station dip times (wait times from CSV)
        self.station_dip_times = station_dip_times or {}

        # Cross-trolley time constant
        self.ct_time = 25.0  # seconds for cross-trolley movement

        # Track previous instruction to avoid duplicate processing
        self.previous_instruction = ""

        # Interlock timer value
        self.interlock_timer_value = 0

        # Debug mode
        self.debug = debug
        self.debug_log = []

    def calculate_travel_time(self, destination: int) -> Tuple[float, bool, int]:
        """Wrapper for distance calculator."""
        result = self.distance_calc.calculate_travel_time(
            self.wagon.current_position, destination, self.wagon.row_no, self.wagon
        )
        return result

    def calculate_travel_time_csharp(self, destination: int) -> Tuple[float, bool, int]:
        """
        C#-style travel time calculation.
        Uses 2-stage distance approach from frmDipTime.cs
        """
        result = self.distance_calc.calculate_travel_time_csharp(
            self.wagon.current_position, destination, self.wagon.row_no, self.wagon
        )
        return result

    def update_position(self, new_position: int):
        """Update the wagon position."""
        self.wagon.current_position = new_position

    def needs_cross_trolley(self, dest_station: int) -> bool:
        """Check if moving to a different row requires cross-trolley."""
        dest_row = self.station_rows.get(dest_station, self.wagon.row_no)
        # Only add CT if wagon changes row AND the current position is also in a different row
        current_row = self.station_rows.get(
            self.wagon.current_position, self.wagon.row_no
        )
        return dest_row != current_row

    def process_instruction(self, instruction: str, value: Any) -> Dict:
        """
        Handles individual auto sequence instructions.
        Returns: dict with updated times and any events
        """
        cmd = instruction.upper().strip()
        result = {"instruction": cmd, "value": value, "time": self.total_time}

        if self.debug:
            self.debug_log.append(
                f"Step: {cmd} {value}, Time before: {self.total_time:.2f}"
            )

        try:
            dest_value = int(value) if value is not None else 0
        except (ValueError, TypeError):
            dest_value = 0

        # GET FROM - Dip Out Time
        if cmd == "GET FROM" or cmd.startswith("GET"):
            # Check if cross-trolley needed
            if self.needs_cross_trolley(dest_value):
                if self.debug:
                    self.debug_log.append(f"  + CT: {self.ct_time}")
                self.total_time += self.ct_time

            if self.wagon.current_position != dest_value:
                travel_time, is_flightbar, censor_dist = self.calculate_travel_time(
                    dest_value
                )
                if self.debug:
                    self.debug_log.append(
                        f"  + Travel ({self.wagon.current_position}->{dest_value}): {travel_time:.2f}"
                    )
                self.total_time += travel_time

            # Add lift time when changing from PUT to GET
            if self.previous_instruction != cmd and self.wagon.lift_speed > 0:
                if self.debug:
                    self.debug_log.append(f"  + Lift: {self.wagon.lift_speed}")
                self.total_time += self.wagon.lift_speed

            if dest_value not in self.dip_data:
                self.dip_data[dest_value] = {
                    "FirstDipInTime": 0,
                    "LastDipOutTime": 0,
                    "DipTime": 0,
                    "FlightBar": False,
                }
            # Track LAST exit time (for multi-cycle sequences)
            self.dip_data[dest_value]["LastDipOutTime"] = self.total_time

        # PUT ON - Dip In Time
        elif cmd == "PUT ON" or cmd.startswith("PUT"):
            # Check if cross-trolley needed
            if self.needs_cross_trolley(dest_value):
                if self.debug:
                    self.debug_log.append(f"  + CT: {self.ct_time}")
                self.total_time += self.ct_time

            if self.wagon.current_position != dest_value:
                travel_time, is_flightbar, censor_dist = self.calculate_travel_time(
                    dest_value
                )
                if self.debug:
                    self.debug_log.append(
                        f"  + Travel ({self.wagon.current_position}->{dest_value}): {travel_time:.2f}"
                    )
                self.total_time += travel_time

            # Add lower down time when changing from GET to PUT
            if self.previous_instruction != cmd and self.wagon.lower_down_speed > 0:
                if self.debug:
                    self.debug_log.append(f"  + Lower: {self.wagon.lower_down_speed}")
                self.total_time += self.wagon.lower_down_speed

            if dest_value not in self.dip_data:
                self.dip_data[dest_value] = {
                    "FirstDipInTime": 0,
                    "LastDipOutTime": 0,
                    "DipTime": 0,
                    "FlightBar": False,
                }
            # Track FIRST entry time (for multi-cycle sequences)
            if self.dip_data[dest_value]["FirstDipInTime"] == 0:
                self.dip_data[dest_value]["FirstDipInTime"] = self.total_time

            # Add station process time (dip_time_sec from CSV) - this is the time wagon stays in tank
            if dest_value in self.station_dip_times:
                process_time = self.station_dip_times[dest_value]
                if process_time > 0:
                    if self.debug:
                        self.debug_log.append(
                            f"  + Process Time (Station {dest_value}): {process_time}"
                        )
                    self.total_time += process_time

        # WAIT FOR SEC / WAIT FOR SECOND
        elif cmd in ["WAIT FOR SEC", "WAIT FOR SECOND", "WAIT SEC"]:
            if self.debug:
                self.debug_log.append(f"  + Wait: {dest_value}")
            self.total_time += float(dest_value)

        # SET INT - Set Interlock
        elif cmd == "SET INT":
            self.interlock_manager.set_interlock(dest_value, self.total_time)

        # WAIT FOR INT / WAIT FOR INTERLOCK
        elif cmd in [
            "WAIT FOR INT",
            "WAIT FOR INTERLOCK",
            "WAIT INTERLOCK",
            "WAIT INT",
        ]:
            # This should be handled by external interlock calculation
            # For now, just pass through
            pass

        # SET CT - Cross Trolley
        elif cmd == "SET CT":
            ct_time = self.cross_trolley.calculate_ct_time(dest_value)
            self.total_time += ct_time

        # WAIT INTERLOCK TIMER
        elif cmd == "WAIT INTERLOCK TIMER":
            if dest_value == self.interlock_timer_value:
                self.total_time += self.interlock_timer_value

        # SET INTERLOCK TIMER
        elif cmd == "SET INTERLOCK TIMER":
            self.interlock_timer_value = dest_value

        # CLAMP / DECLAMP / TILT / UNTILT
        elif cmd in ["CLAMP", "DECLAMP", "TILT", "UNTILT"]:
            self.total_time += float(dest_value)
            self.previous_instruction = cmd

        # Update position after processing
        if cmd in ["GET FROM", "PUT ON"]:
            self.update_position(dest_value)

        if self.debug:
            self.debug_log.append(f"  -> Time after: {self.total_time:.2f}")

        return result

    def finalize_dip_times(self) -> Dict:
        """
        Calculates final dip times.

        Returns first DipIn and last DipOut for each station.
        Does NOT automatically wrap to max_cycle_time.
        """
        results = {}

        for station_no, times in self.dip_data.items():
            first_dip_in = times.get("FirstDipInTime", 0)
            last_dip_out = times.get("LastDipOutTime", 0)

            # Handle case where only DipIn exists (end of sequence) - don't wrap!
            if first_dip_in > 0 and last_dip_out == 0:
                # Last PUT ON without GET - leave as is (no wrap)
                last_dip_out = 0

            # Handle case where DipIn > DipOut (cyclic sequence) - wrap
            if first_dip_in > last_dip_out and first_dip_in > 0 and last_dip_out > 0:
                last_dip_out += self.max_cycle_time

            dip_time = last_dip_out - first_dip_in

            # Actual Dip Time Scaling
            if self.max_cycle_time > 0:
                dip_time_actual = (
                    dip_time * self.actual_cycle_time
                ) / self.max_cycle_time
            else:
                dip_time_actual = dip_time

            results[station_no] = {
                "DipInTime": round(first_dip_in, 2),
                "DipOutTime": round(last_dip_out, 2) if last_dip_out > 0 else 0,
                "DipTime": round(dip_time, 2) if dip_time > 0 else 0,
                "DipTimeActual": round(dip_time_actual, 2)
                if dip_time_actual > 0
                else 0,
                "FlightBar": times.get("FlightBar", False),
            }

        return results


class DipTimeProcessor:
    """
    C#-style Dip Time Processor.
    Mimics the DipTimeCalculation() from frmDipTime.cs

    Simplified version that uses the C# 2-stage distance calculation.
    """

    def __init__(self, wagon_id: str, wagon_config: Dict):
        self.wagon_id = wagon_id
        self.wagon_config = wagon_config

        # Current state
        self.currentposition = wagon_config.get("basic_pos", 0)
        self.basicposition = wagon_config.get("basic_pos", 0)
        self.rowno = wagon_config.get("row", 1)

        # Speed parameters
        self.sfspeed = wagon_config.get("sf", 30)
        self.fspeed = wagon_config.get("f", 20)
        self.sspeed = wagon_config.get("s", 10)
        self.liftspeed = wagon_config.get("lift_time", 0)
        self.lowerspeed = wagon_config.get("lower_time", 0)

        # Tracking
        self.timevalue = 0.0
        self.previous_instruction = ""
        self.dip_data = {}  # station → {DipInTime, DipOutTime, FlightBar}

    def process_instruction(self, cmd: str, value: Any) -> float:
        """Process single instruction like C# DipTimeCalculation()"""
        cmd_upper = cmd.upper().strip()

        try:
            dest_value = int(value)
        except (ValueError, TypeError):
            dest_value = 0

        if cmd_upper in ["GET FROM", "PUT ON"]:
            # Calculate travel time using C# 2-stage approach
            if self.currentposition != dest_value:
                travel_time = calculate_time_csharp(
                    self.currentposition,
                    dest_value,
                    self.rowno,
                    self.sfspeed,
                    self.fspeed,
                    self.sspeed,
                )
                self.timevalue += travel_time

            # Add lift/lower only on instruction change (matching C#)
            if self.previous_instruction != cmd_upper:
                if cmd_upper.startswith("GET"):
                    if self.liftspeed > 0:
                        self.timevalue += self.liftspeed
                elif cmd_upper.startswith("PUT"):
                    if self.lowerspeed > 0:
                        self.timevalue += self.lowerspeed

            # Track dip in/out times
            if cmd_upper == "PUT ON":
                if dest_value not in self.dip_data:
                    self.dip_data[dest_value] = {
                        "DipInTime": 0,
                        "DipOutTime": 0,
                        "FlightBar": False,
                        "Load": "L0",
                    }
                self.dip_data[dest_value]["DipInTime"] = self.timevalue

            elif cmd_upper == "GET FROM":
                if dest_value in self.dip_data:
                    self.dip_data[dest_value]["DipOutTime"] = self.timevalue

            # Update position
            self.currentposition = dest_value

        elif cmd_upper in ["WAIT FOR SEC", "WAIT FOR SECOND"]:
            self.timevalue += float(dest_value)

        elif cmd_upper in ["CLAMP", "DECLAMP", "TILT", "UNTILT"]:
            self.timevalue += float(dest_value)

        elif cmd_upper == "SET CT":
            ct_speed = 10  # m/min default
            ct_distance = 2230  # mm
            ct_time = ct_distance / (ct_speed * 16.66)
            self.timevalue += ct_time

        self.previous_instruction = cmd_upper
        return self.timevalue

    def get_total_time(self) -> float:
        """Get total accumulated time"""
        return self.timevalue

    def calculate_dip_times(
        self, max_cycle_time: float = 600, actual_cycle_time: float = 600
    ) -> Dict:
        """
        Calculate final dip times with cycle scaling.
        Mimics C# logic for DipTimeActual calculation.
        """
        results = {}

        for stn, data in self.dip_data.items():
            dip_in = data.get("DipInTime", 0)
            dip_out = data.get("DipOutTime", 0)

            # Handle cycle wrap (matching C# logic)
            if dip_in > dip_out and dip_in > 0 and dip_out > 0:
                dip_out += max_cycle_time

            dip_time = dip_out - dip_in if dip_out > dip_in else 0

            # Calculate actual dip time with cycle scaling
            if max_cycle_time > 0:
                dip_time_actual = (dip_time * actual_cycle_time) / max_cycle_time
            else:
                dip_time_actual = dip_time

            results[stn] = {
                "DipInTime": round(dip_in, 2),
                "DipOutTime": round(dip_out, 2) if dip_out > 0 else 0,
                "DipTime": round(dip_time, 2),
                "DipTimeActual": round(dip_time_actual, 2),
                "FlightBar": data.get("FlightBar", False),
                "Load": data.get("Load", "L0"),
            }

        return results

    def print_debug_log(self):
        """Print the debug log."""
        print("\n" + "=" * 60)
        print("DEBUG LOG - Step by Step Calculation")
        print("=" * 60)
        for line in self.debug_log:
            print(line)
        print("=" * 60)

    def get_all_events(self) -> List[Dict]:
        """
        Get all dip events in chronological order.
        Returns list of {station, event_type, time}
        """
        events = []
        for station_no, times in self.dip_data.items():
            if times.get("FirstDipInTime", 0) > 0:
                events.append(
                    {
                        "station": station_no,
                        "type": "IN",
                        "time": times.get("FirstDipInTime", 0),
                    }
                )
            if times.get("LastDipOutTime", 0) > 0:
                events.append(
                    {
                        "station": station_no,
                        "type": "OUT",
                        "time": times.get("LastDipOutTime", 0),
                    }
                )
        return sorted(events, key=lambda x: x["time"])


class MultiTankProcessor:
    """
    Handles multi-tank/multi-row dip time calculations.
    Processes multiple wagons and manages inter-tank operations.
    """

    def __init__(
        self,
        max_cycle_time: float,
        actual_cycle_time: float,
        distance_data: Dict = None,
        ct_data: Dict = None,
    ):
        self.max_cycle_time = max_cycle_time
        self.actual_cycle_time = actual_cycle_time
        self.distance_data = distance_data or {}
        self.ct_data = ct_data or {}

        # Store all wagon processors
        self.wagon_processors = {}

        # Interlock events storage
        self.interlock_events = []  # (time, wagon_id, interlock_no, instruction)

        # Final dip times aggregated across all wagons
        self.aggregated_dip_times = {}

    def add_wagon(
        self, wagon_id: str, wagon_params: Dict, station_rows: Dict[int, int] = None
    ):
        """Add a wagon to the processor."""
        wagon = Wagon(
            wagon_id=wagon_id,
            basic_pos=wagon_params.get("BasicPosition", 0),
            row_no=wagon_params.get("RowNo", 1),
            sf_speed=wagon_params.get("SuperfastSpeed", 1),
            f_speed=wagon_params.get("FastSpeed", 1),
            s_speed=wagon_params.get("SlowSpeed", 1),
            lift_speed=wagon_params.get("LiftSpeed", 0),
            lower_speed=wagon_params.get("LowerDownSpeed", 0),
        )

        distance_calc = DistanceCalculator(self.distance_data)
        interlock_mgr = InterlockManager()
        cross_trolley = CrossTrolley(self.ct_data)

        processor = SequenceProcessor(
            wagon,
            self.max_cycle_time,
            self.actual_cycle_time,
            distance_calc,
            interlock_mgr,
            cross_trolley,
            station_rows,
        )

        self.wagon_processors[wagon_id] = processor
        return processor

    def process_sequence(self, wagon_id: str, instructions: List[Tuple[str, Any]]):
        """Process a sequence of instructions for a specific wagon."""
        if wagon_id not in self.wagon_processors:
            raise ValueError(f"Wagon {wagon_id} not added. Call add_wagon first.")

        processor = self.wagon_processors[wagon_id]

        for cmd, val in instructions:
            result = processor.process_instruction(cmd, val)

            # Track interlock events
            if cmd.upper() in ["SET INT", "WAIT FOR INT"]:
                self.interlock_events.append((processor.total_time, wagon_id, val, cmd))

    def calculate_interlock_wait(
        self,
        wagon_id: str,
        interlock_no: int,
        other_wagon_instructions: List[Tuple[str, Any]],
    ) -> float:
        """
        Calculate wait time for an interlock.
        Returns the time the other wagon sets the interlock.
        """
        # Find SET INT event for this interlock
        for time, w_id, int_no, instr in self.interlock_events:
            if instr.upper() == "SET INT" and int_no == interlock_no:
                return time
        return 0.0

    def get_dip_times(self) -> Dict:
        """Get aggregated dip times from all wagons."""
        all_dip_times = {}

        for wagon_id, processor in self.wagon_processors.items():
            wagon_dip_times = processor.finalize_dip_times()

            for station_no, times in wagon_dip_times.items():
                if station_no not in all_dip_times:
                    all_dip_times[station_no] = {
                        "Wagon": wagon_id,
                        "DipInTime": times["DipInTime"],
                        "DipOutTime": times["DipOutTime"],
                        "DipTime": times["DipTime"],
                        "DipTimeActual": times["DipTimeActual"],
                        "FlightBar": times["FlightBar"],
                    }

        return all_dip_times

    def get_wagon_dip_times(self, wagon_id: str) -> Dict:
        """Get dip times for a specific wagon."""
        if wagon_id not in self.wagon_processors:
            return {}
        return self.wagon_processors[wagon_id].finalize_dip_times()


def calculate_wagon_dip_times(
    wagon_params: Dict,
    instructions: List[Tuple[str, Any]],
    max_cycle_time: float,
    actual_cycle_time: float,
    distance_data: Dict = None,
    ct_data: Dict = None,
    station_rows: Dict[int, int] = None,
    station_positions: Dict[int, int] = None,
    station_dip_times: Dict[int, int] = None,
) -> Dict:
    """
    Main entry point for single wagon DIP time calculation.

    Args:
        wagon_params (dict): Dictionary with wagon speed and position settings
        instructions (list): List of (command, value) tuples
        max_cycle_time (float): The design cycle time
        actual_cycle_time (float): The operational cycle time
        distance_data (dict): Optional distance data for zone calculations
        ct_data (dict): Optional cross-trolley data
        station_rows (dict): Optional station -> row mapping
        station_positions (dict): Optional station -> position mapping
        station_dip_times (dict): Optional station -> expected dip time (wait time)

    Returns:
        dict: Station-wise dip times
    """
    wagon = Wagon(
        wagon_id=wagon_params.get("WagonId", "Wagon1"),
        basic_pos=wagon_params.get("BasicPosition", 0),
        row_no=wagon_params.get("RowNo", 1),
        sf_speed=wagon_params.get("SuperfastSpeed", 1),
        f_speed=wagon_params.get("FastSpeed", 1),
        s_speed=wagon_params.get("SlowSpeed", 1),
        lift_speed=wagon_params.get("LiftSpeed", 0),
        lower_speed=wagon_params.get("LowerDownSpeed", 0),
    )

    distance_calc = DistanceCalculator(distance_data or {}, station_positions)
    interlock_mgr = InterlockManager()
    cross_trolley = CrossTrolley(ct_data or {})

    processor = SequenceProcessor(
        wagon,
        max_cycle_time,
        actual_cycle_time,
        distance_calc,
        interlock_mgr,
        cross_trolley,
        station_rows,
        station_dip_times,
    )

    for cmd, val in instructions:
        processor.process_instruction(cmd, val)

    return processor.finalize_dip_times()


def calculate_multi_tank_dip_times(
    wagons_data: Dict[str, Dict],
    sequences: Dict[str, List[Tuple[str, Any]]],
    max_cycle_time: float,
    actual_cycle_time: float,
    distance_data: Dict = None,
    ct_data: Dict = None,
) -> Dict:
    """
    Main entry point for multi-tank DIP time calculation.

    Args:
        wagons_data: Dict of wagon_id -> wagon_params
        sequences: Dict of wagon_id -> list of (command, value) tuples
        max_cycle_time: The design cycle time
        actual_cycle_time: The operational cycle time
        distance_data: Optional distance data for zone calculations
        ct_data: Optional cross-trolley data

    Returns:
        dict: Station-wise dip times for all wagons
    """
    processor = MultiTankProcessor(
        max_cycle_time, actual_cycle_time, distance_data, ct_data
    )

    # Add all wagons
    for wagon_id, params in wagons_data.items():
        processor.add_wagon(wagon_id, params)

    # Process sequences for all wagons
    for wagon_id, instructions in sequences.items():
        processor.process_sequence(wagon_id, instructions)

    return processor.get_dip_times()


# Example test function
def run_test():
    """Test the dip time calculation with example data."""
    example_wagon = {
        "WagonId": "Wagon1",
        "BasicPosition": 0,
        "RowNo": 1,
        "SuperfastSpeed": 30,
        "FastSpeed": 20,
        "SlowSpeed": 10,
        "LiftSpeed": 5,
        "LowerDownSpeed": 5,
    }

    example_sequence = [
        ("GET FROM", 100),
        ("PUT ON", 1000),
        ("WAIT FOR SEC", 30),
        ("GET FROM", 1000),
        ("PUT ON", 2000),
        ("WAIT FOR SEC", 20),
        ("GET FROM", 2000),
        ("PUT ON", 100),
    ]

    result = calculate_wagon_dip_times(example_wagon, example_sequence, 300, 300)
    print("Single Wagon Dip Time Result:")
    import json

    print(json.dumps(result, indent=2))

    # Test multi-tank
    wagons = {
        "Wagon1": {**example_wagon, "BasicPosition": 0},
        "Wagon2": {**example_wagon, "BasicPosition": 0, "WagonId": "Wagon2"},
    }

    sequences = {
        "Wagon1": example_sequence,
        "Wagon2": [
            ("GET FROM", 50),
            ("PUT ON", 500),
            ("WAIT FOR SEC", 25),
            ("GET FROM", 500),
            ("PUT ON", 100),
        ],
    }

    multi_result = calculate_multi_tank_dip_times(wagons, sequences, 300, 300)
    print("\nMulti-Tank Dip Time Result:")
    print(json.dumps(multi_result, indent=2))


if __name__ == "__main__":
    run_test()
