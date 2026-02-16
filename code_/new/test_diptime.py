from diptime_calculation import calculate_wagon_dip_times
import json

def test_basic_logic():
    print("Running test_basic_logic...")
    wagon_params = {
        "BasicPosition": 0,
        "SuperfastSpeed": 30,
        "FastSpeed": 20,
        "SlowSpeed": 10,
        "LiftSpeed": 5,
        "LowerDownSpeed": 5
    }
    
    # Distance = 1000
    # Slow (500) -> 500 / (10 * 16.66) = 3.001
    # Fast (500) -> 500 / (20 * 16.66) = 1.500
    # Total travel = 4.501
    
    sequence = [
        ("PUT ON", 1000),      # Travel (4.5) + Lower (5) = 9.5. Dip In = 9.5
        ("WAIT FOR SEC", 10),   # Total = 19.5
        ("GET FROM", 1000),    # Travel (0) + Lift (5) = 24.5. Dip Out = 24.5
    ]
    
    # Dip Time = 24.5 - 9.5 = 15.0
    
    res = calculate_wagon_dip_times(wagon_params, sequence, 100, 100)
    print(json.dumps(res, indent=2))
    
    assert 1000 in res
    assert abs(res[1000]["DipTime"] - 15.0) < 0.1
    print("test_basic_logic passed!\n")

def test_cycle_scaling():
    print("Running test_cycle_scaling...")
    wagon_params = {
        "BasicPosition": 0,
        "SuperfastSpeed": 30,
        "FastSpeed": 20,
        "SlowSpeed": 10,
        "LiftSpeed": 0,
        "LowerDownSpeed": 0
    }
    
    sequence = [
        ("PUT ON", 0),       # Dip In = 0
        ("WAIT FOR SEC", 50), # Total = 50
        ("GET FROM", 0),      # Dip Out = 50
    ]
    
    # Dip Time = 50
    # Scale: (50 * 600) / 300 = 100
    
    res = calculate_wagon_dip_times(wagon_params, sequence, 300, 600)
    print(json.dumps(res, indent=2))
    
    assert res[0]["DipTime"] == 100.0
    print("test_cycle_scaling passed!\n")

def test_cycle_overflow():
    print("Running test_cycle_overflow...")
    wagon_params = {
        "BasicPosition": 0,
        "SuperfastSpeed": 30,
        "FastSpeed": 20,
        "SlowSpeed": 10,
        "LiftSpeed": 0,
        "LowerDownSpeed": 0
    }
    
    # Let's simulate a case where DipIn > DipOut by using multiple stations
    # If station 2 is Dip In, and then some other work happens, and then station 2 is Dip Out
    # But for this simple logic, total_time only increases.
    # The requirement says: "If DipInTime > DipOutTime: DipOutTime += max_cycle_time"
    # This usually happens in circular sequences where DipIn happens at the end of cycle 1 
    # and DipOut happens at the beginning of cycle 2.
    
    # To test this, I'll manually trigger the condition by mocking the data if needed, 
    # but I'll see if I can construct a sequence.
    # Actually, the current processor only calculates linear time. 
    # The overflow check is for when the user provides data from different cycles.
    
    # Let's verify the logic in finalize_dip_times works if we simulate the state
    from diptime_calculation import SequenceProcessor, Wagon
    w = Wagon("W1", 0, 1, 1, 1, 0, 0)
    sp = SequenceProcessor(w, 300, 300)
    sp.dip_data[5] = {"DipInTime": 250, "DipOutTime": 50, "DipTime": 0}
    
    res = sp.finalize_dip_times()
    # dip_out should become 50 + 300 = 350
    # dip_time = 350 - 250 = 100
    print(json.dumps(res, indent=2))
    assert res[5]["DipTime"] == 100.0
    print("test_cycle_overflow logic verified!\n")

if __name__ == "__main__":
    test_basic_logic()
    test_cycle_scaling()
    test_cycle_overflow()
    print("All tests passed successfully!")
