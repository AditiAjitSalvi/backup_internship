from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import uuid
import time

class StationType(Enum):
    LOADING = "Loading"
    UNLOADING = "Unloading"
    TANK = "Tank"
    BUFFER = "Buffer"
    CROSS_TROLLEY_INTERFACE = "CrossTrolleyInterface"

@dataclass
class Process:
    name: str
    duration: float  # seconds
    is_completed: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None

@dataclass
class Load:
    load_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    processes: List[Process] = field(default_factory=list)
    current_station_id: Optional[str] = None
    next_process_index: int = 0

    @property
    def current_process(self) -> Optional[Process]:
        if 0 <= self.next_process_index < len(self.processes):
            return self.processes[self.next_process_index]
        return None

    def complete_current_process(self):
        cp = self.current_process
        if cp:
            cp.is_completed = True
            cp.end_time = time.time()
            self.next_process_index += 1

@dataclass
class Station:
    station_id: str
    station_type: StationType
    position_mm: float
    compatible_processes: List[str] = field(default_factory=list)
    current_load_id: Optional[str] = None

    def is_available(self) -> bool:
        return self.current_load_id is None

@dataclass
class Transporter:
    transporter_id: str
    position_mm: float = 0.0
    speed_mm_s: float = 500.0  # default speed
    lift_time: float = 5.0
    lower_time: float = 5.0
    current_load_id: Optional[str] = None
    row_id: Optional[str] = None

    def is_available(self) -> bool:
        return self.current_load_id is None

@dataclass
class Row:
    row_id: str
    stations: List[Station] = field(default_factory=list)
    transporter: Optional[Transporter] = None

    def get_station_by_id(self, station_id: str) -> Optional[Station]:
        for s in self.stations:
            if s.station_id == station_id:
                return s
        return None

@dataclass
class CrossTrolley:
    trolley_id: str
    connected_rows: List[str] = field(default_factory=list) # List of row_ids
    position_row_id: Optional[str] = None # Current row it's aligned with
    speed_row_s: float = 1.0 # time to move between adjacent rows
    current_load_id: Optional[str] = None

    def is_available(self) -> bool:
        return self.current_load_id is None

@dataclass
class Plant:
    rows: Dict[str, Row] = field(default_factory=dict)
    cross_trolleys: List[CrossTrolley] = field(default_factory=list)
    loads: Dict[str, Load] = field(default_factory=dict)
