# Dip Time Calculation - Changes Documentation

## C# Travel Time Formula (Exact)

The C# code uses a database function `getDistance()` to calculate travel time:

```csharp
// 1. Get distance data from current position to destination+1 or destination-1
dtDistance1 = getDistance(currentposition, destinationValue + 1, rowno)

// 2. Get distance data from destination+1 or destination-1 to destination
dtDistance2 = getDistance(destinationValue + 1, destinationValue, rowno)

// 3. Calculate zones
d1 = dtDistance1.Rows[0]["Distance"]    // Superfast zone
d2 = dtDistance1.Rows[1]["Distance"]      // Fast zone
d3 = dtDistance2.Rows[0]["Distance"]    // Slow zone part 1
d4 = dtDistance2.Rows[1]["Distance"]    // Slow zone part 2

distance1 = |d1 - d2|     // Superfast zone distance
distance2 = |d3 - d4| - CensorDistance  // Fast zone distance
distance3 = CensorDistance  // Slow zone (default 50mm)

// 4. Calculate time
time = (distance1 / (sfspeed * 16.66)) + 
        (distance2 / (fspeed * 16.66)) + 
        (distance3 / (sspeed * 16.66))
```

## Key Parameters from wagon_config.csv

| Parameter | Value |
|-----------|-------|
| Superfast Speed | 12 mm/s |
| Fast Speed | 25 mm/s |
| Slow Speed | 5 mm/s |
| Lift Time | 12 seconds |
| Lower Time | 12 seconds |
| Basic Position | 1 |

## Key Parameters from stations CSV

| Station | Position (mm) | Row | Dip Time (sec) |
|---------|---------------|-----|----------------|
| 1 | 0 | 1 | 0 |
| 2 | 2000 | 1 | 180 |
| 3 | 3000 | 1 | 180 |
| 4 | 4000 | 1 | 180 |
| 5 | 5000 | 1 | 60 |
| 6 | 6000 | 2 | 20 |
| 7 | 2300 | 2 | 45 |
| 8 | 3100 | 2 | 90 |

## Python Implementation Changes

### Files Modified:

1. **diptime_calculation.py** - Main calculation engine
2. **industrial_diptime_system.py** - CSV integration
3. **config.py** - Configuration parameters
4. **debug_trace.py** - Debug script

### Key Implementation Details:

1. **DistanceCalculator class** - Handles zone-based travel calculation
2. **SequenceProcessor class** - Processes GET FROM / PUT ON instructions
3. **Wagon class** - Stores wagon parameters

### Travel Time Formula (Python)

```python
def calculate_travel_time(self, from_pos, to_pos, row_no, wagon):
    # Zone distribution:
    # - Slow zone: min(distance, 50) mm
    # - Fast zone: min(remaining, 1000) mm
    # - Superfast zone: remaining after slow and fast
    
    distance = abs(to_pos - from_pos)
    d3 = min(distance, 50)  # Slow zone (sensor)
    remaining = distance - d3
    d2 = min(remaining, 1000)  # Fast zone
    d1 = remaining - d2  # Superfast zone
    
    time = d1/(sfspeed*16.66) + d2/(fspeed*16.66) + d3/(sspeed*16.66)
    return time
```

### Key Features:

1. **Cross-Trolley (CT) Detection**: Detects when wagon moves between rows
2. **Multi-tank Support**: Handles different rows
3. **First In / Last Out Tracking**: Tracks first entry and last exit times
4. **Process Time Integration**: Can include dip_time_sec from stations CSV
5. **Debug Mode**: Step-by-step calculation tracing

## How to Use

### Run Normal Calculation:
```bash
cd C:\Users\aditi\Downloads\multitank1solu\new\diptime
python industrial_diptime_system.py
```

### Run Debug Trace:
```bash
python debug_trace.py
```

### Configuration:

Edit `config.py` to adjust:
- SUPERFAST_SPEED
- FAST_SPEED
- SLOW_SPEED
- LIFT_SPEED
- LOWER_SPEED
- CT_TIME

## Known Differences from Expected Output

The calculated values may differ from expected due to:
1. Database distance data (getDistance function) - not simulated
2. Exact CT timing between rows
3. Different travel formula in database

## To Match Expected Output

The expected output shows values WITHOUT process times in DipIn/DipOut:
- DipIn/DipOut = travel + lift/lower only
- DipTime = calculated from DipOut - DipIn

If your expected includes different values, check:
1. The database getDistance() function
2. CT time between row 1 and row 2
3. Basic position handling
