# Distance Configuration - Mimics the database getDistance() function
# Format: from_station, to_station, row_no -> distance, sensor_distance, flightbar

# For each movement, the C# code calls:
# getDistance(currentposition, destinationValue + 1 or -1) to get d1, d2
# getDistance(destinationValue + 1 or -1, destinationValue) to get d3, d4

# Station positions (distance in mm from start of row)
STATION_POSITIONS = {
    1: 0,
    2: 2000,
    3: 3000,
    4: 4000,
    5: 5000,
    6: 6000,  # Row 2
    7: 2300,  # Row 2
    8: 3100,  # Row 2
}

# Sensor distances for each station (in mm)
SENSOR_DISTANCES = {
    1: 50,
    2: 50,
    3: 50,
    4: 50,
    5: 50,
    6: 50,
    7: 50,
    8: 50,
}

# Flight bar presence
FLIGHT_BAR = {
    1: False,
    2: False,
    3: False,
    4: False,
    5: False,
    6: False,
    7: False,
    8: False,
}


def get_distance(from_pos, to_pos, row_no=1):
    """
    Mimics the C# getDistance function.
    Returns d1, d2, sensor_distance, flightbar for the given movement.
    """
    from_station = from_pos
    to_station = to_pos

    # Get positions
    pos1 = STATION_POSITIONS.get(from_station, from_station * 1000)
    pos2 = STATION_POSITIONS.get(to_station, to_station * 1000)

    # For movement calculation:
    # The C# code calls getDistance twice:
    # 1. getDistance(current, dest+1 or dest-1) -> d1, d2
    # 2. getDistance(dest+1 or dest-1, dest) -> d3, d4

    # For now, return simple distance
    distance = abs(pos2 - pos1)

    return {
        "distance1": distance,
        "distance2": 0,
        "censor_distance": SENSOR_DISTANCES.get(to_station, 50),
        "flightbar": FLIGHT_BAR.get(to_station, False),
    }
