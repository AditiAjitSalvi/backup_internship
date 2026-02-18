import math
from typing import List, Optional, Tuple
from ..domain.models import Plant, Row, Station, Load, StationType, Transporter, CrossTrolley

class SchedulingEngine:
    def __init__(self, plant: Plant):
        self.plant = plant
        self.timing_buffer = 0.1 # 10% safety buffer

    def calculate_traversal_time(self, transporter: Transporter, target_pos_mm: float) -> float:
        distance = abs(target_pos_mm - transporter.position_mm)
        # Simplified physics for now (as per requirements)
        # transporter.speed_mm_s * 16.66 in the original code seems to be a conversion factor
        # but here we'll use a direct speed for clarity.
        time_needed = distance / transporter.speed_mm_s
        return time_needed * (1 + self.timing_buffer)

    def find_target_station(self, load: Load, current_row_id: str) -> Optional[Tuple[str, str]]:
        """
        Returns (row_id, station_id) for the current process of the load.
        """
        process = load.current_process
        if not process:
            return None

        # 1. Search in current row
        row = self.plant.rows.get(current_row_id)
        if row:
            target = self._search_row_for_station(row, process.name)
            if target:
                return current_row_id, target.station_id

        # 2. Search in other rows (Cross-row logic)
        for row_id, row in self.plant.rows.items():
            if row_id == current_row_id:
                continue
            target = self._search_row_for_station(row, process.name)
            if target:
                return row_id, target.station_id

        # 3. Fallback to Buffer in current row if tank unavailable
        if row:
            for s in row.stations:
                if s.station_type == StationType.BUFFER and s.is_available():
                    return current_row_id, s.station_id

        return None

    def _search_row_for_station(self, row: Row, process_name: str) -> Optional[Station]:
        # Capacity check (row should have a transporter)
        if not row.transporter:
            return None
        
        for s in row.stations:
            # Check if station supports the process name
            if process_name in s.compatible_processes:
                if s.is_available():
                    return s
        return None

    def get_transfer_route(self, load: Load, from_row_id: str, to_row_id: str) -> List[dict]:
        """
        Calculates moves for transporter -> crossTrolley -> transporter.
        """
        if from_row_id == to_row_id:
            return []

        # Find interface stations in both rows
        from_row = self.plant.rows[from_row_id]
        to_row = self.plant.rows[to_row_id]
        
        from_interface = next((s for s in from_row.stations if s.station_type == StationType.CROSS_TROLLEY_INTERFACE), None)
        to_interface = next((s for s in to_row.stations if s.station_type == StationType.CROSS_TROLLEY_INTERFACE), None)

        if not from_interface or not to_interface:
            return []

        # Find available trolley
        trolley = next((ct for ct in self.plant.cross_trolleys if ct.is_available()), None)
        if not trolley:
            return []

        # Route consists of steps for the plant manager to execute
        route = [
            {"type": "TRANS_MOVE", "row": from_row_id, "to_stn": from_interface.station_id},
            {"type": "TROLLEY_PICKUP", "trolley": trolley.trolley_id, "from_row": from_row_id},
            {"type": "TROLLEY_MOVE", "trolley": trolley.trolley_id, "to_row": to_row_id},
            {"type": "TROLLEY_DROPOFF", "trolley": trolley.trolley_id, "to_row": to_row_id},
            {"type": "TRANS_PICKUP", "row": to_row_id, "from_stn": to_interface.station_id}
        ]
        return route
