import math

class Wagon:
    """Represents an industrial wagon with its speed parameters and position."""
    def __init__(self, wagon_id, basic_pos, sf_speed, f_speed, s_speed, lift_speed, lower_speed):
        self.wagon_id = wagon_id
        self.basic_position = basic_pos
        self.superfast_speed = sf_speed
        self.fast_speed = f_speed
        self.slow_speed = s_speed
        self.lift_speed = lift_speed
        self.lower_down_speed = lower_speed
        self.current_position = basic_pos

class SequenceProcessor:
    """Processes a sequence of instructions to calculate station-wise dip times."""
    def __init__(self, wagon, max_cycle_time, actual_cycle_time):
        self.wagon = wagon
        self.max_cycle_time = max_cycle_time
        self.actual_cycle_time = actual_cycle_time
        self.total_time = 0.0
        self.dip_data = {} # station_no -> {DipInTime, DipOutTime, DipTime}

    def calculate_travel_time(self, destination):
        """Calculates travel time using the segmented zone formula."""
        if self.wagon.current_position == destination:
            return 0.0
        
        distance_total = abs(destination - self.wagon.current_position)
        
        # Defining zones based on standard approach:
        # Slow zone: last 500 units
        # Fast zone: next 1000 units
        # Superfast zone: any remaining distance
        
        d3 = min(distance_total, 500.0)
        remaining = distance_total - d3
        
        d2 = min(remaining, 1000.0)
        d1 = remaining - d2
        
        # Travel Time Formula:
        # (distance1 / (SuperfastSpeed * 16.66)) +
        # (distance2 / (FastSpeed * 16.66)) +
        # (distance3 / (SlowSpeed * 16.66))
        
        t1 = d1 / (self.wagon.superfast_speed * 16.66) if self.wagon.superfast_speed > 0 else 0
        t2 = d2 / (self.wagon.fast_speed * 16.66) if self.wagon.fast_speed > 0 else 0
        t3 = d3 / (self.wagon.slow_speed * 16.66) if self.wagon.slow_speed > 0 else 0
        
        travel_time = t1 + t2 + t3
        self.wagon.current_position = destination
        return travel_time

    def process_instruction(self, instruction, value):
        """Handles individual auto sequence instructions."""
        cmd = instruction.upper()
        
        if cmd == "GET FROM":
            # Dip Out Time
            travel_time = self.calculate_travel_time(value)
            self.total_time += travel_time + self.wagon.lift_speed
            
            if value not in self.dip_data:
                self.dip_data[value] = {"DipInTime": 0, "DipOutTime": 0, "DipTime": 0}
            self.dip_data[value]["DipOutTime"] = self.total_time
            
        elif cmd == "PUT ON":
            # Dip In Time
            travel_time = self.calculate_travel_time(value)
            self.total_time += travel_time + self.wagon.lower_down_speed
            
            if value not in self.dip_data:
                self.dip_data[value] = {"DipInTime": 0, "DipOutTime": 0, "DipTime": 0}
            self.dip_data[value]["DipInTime"] = self.total_time
            
        elif cmd in ["WAIT FOR SEC", "WAIT FOR SECOND", "CLAMP", "DECLAMP", "TILT", "UNTILT"]:
            # Simple time delay instructions
            self.total_time += float(value)
            
        elif cmd in ["WAIT FOR INT", "SET INT", "SET CT"]:
            # Interlock and cross trolley movement (assuming no extra time for logic)
            pass

    def finalize_dip_times(self):
        """Calculates final dip times with cycle handling and scaling."""
        results = {}
        for station_no, times in self.dip_data.items():
            dip_in = times["DipInTime"]
            dip_out = times["DipOutTime"]
            
            # Logic from requirements:
            # If DipInTime > DipOutTime: DipOutTime += max_cycle_time
            if dip_in > dip_out and dip_in > 0 and dip_out > 0:
                dip_out += self.max_cycle_time
                
            dip_time = dip_out - dip_in
            
            # Actual Dip Time Scaling:
            # DipTimeActual = (DipTime * ActualCycleTime) / max_cycle_time
            if self.max_cycle_time > 0:
                dip_time_actual = (dip_time * self.actual_cycle_time) / self.max_cycle_time
            else:
                dip_time_actual = dip_time
                
            results[station_no] = {
                "DipInTime": round(dip_in, 2),
                "DipOutTime": round(dip_out, 2),
                "DipTime": round(dip_time_actual, 2)
            }
        return results

def calculate_wagon_dip_times(wagon_params, instructions, max_cycle_time, actual_cycle_time):
    """
    Main entry point for DIP time calculation.
    
    Args:
        wagon_params (dict): Dictionary with wagon speed and position settings.
        instructions (list): List of (command, value) tuples.
        max_cycle_time (float): The design cycle time.
        actual_cycle_time (float): The operational cycle time.
        
    Returns:
        dict: Station-wise dip times.
    """
    wagon = Wagon(
        wagon_id="Wagon1",
        basic_pos=wagon_params.get("BasicPosition", 0),
        sf_speed=wagon_params.get("SuperfastSpeed", 1),
        f_speed=wagon_params.get("FastSpeed", 1),
        s_speed=wagon_params.get("SlowSpeed", 1),
        lift_speed=wagon_params.get("LiftSpeed", 0),
        lower_speed=wagon_params.get("LowerDownSpeed", 0)
    )
    
    processor = SequenceProcessor(wagon, max_cycle_time, actual_cycle_time)
    
    for cmd, val in instructions:
        processor.process_instruction(cmd, val)
        
    return processor.finalize_dip_times()

if __name__ == "__main__":
    # Example usage for demonstration
    example_wagon = {
        "BasicPosition": 0,
        "SuperfastSpeed": 30,
        "FastSpeed": 20,
        "SlowSpeed": 10,
        "LiftSpeed": 5,
        "LowerDownSpeed": 5
    }
    
    example_sequence = [
        ("GET FROM", 100),   # Initial fetch
        ("PUT ON", 1000),    # Put into station 1000 (Dip In)
        ("WAIT FOR SEC", 30), # Waiting in tank
        ("GET FROM", 1000),  # Fetch from station 1000 (Dip Out)
        ("PUT ON", 2000)     # Move to next
    ]
    
    res = calculate_wagon_dip_times(example_wagon, example_sequence, 300, 300)
    print("Dip Time Calculation Result:")
    import json
    print(json.dumps(res, indent=2))
