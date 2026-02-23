import json
import os
import threading
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from ..domain.models import Plant, Load, Row, Transporter, Station, StationType, Process
from ..scheduler.engine import SchedulingEngine

class PlantManager:
    def __init__(self, config_path: str = "plant_config.json"):
        self.config_path = config_path
        self.plant = Plant()
        self.scheduler = SchedulingEngine(self.plant)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.locks: Dict[str, threading.Lock] = {} # Row-level locks
        self.pending_loads: List[str] = []
        self.active_processes: Dict[str, str] = {} # load_id -> current_process_name
        self.event_logs: List[dict] = [] # History of all events
        self._load_config()

    def _load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                # Restore state logic (simplified for implementation example)
                # In a real app, this would reconstruct the Plant object
                print(f"Restoring plant state from {self.config_path}")
        else:
            print("No existing configuration found. Starting fresh.")

    def save_state(self):
        # Basic persistence of load status and plant config
        state = {
            "loads": {lid: {"pidx": l.next_process_index, "stn": l.current_station_id} for lid, l in self.plant.loads.items()},
            # ... other plant state
        }
        with open(self.config_path, 'w') as f:
            json.dump(state, f, indent=4)

    def record_event(self, event_type: str, details: dict):
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        self.event_logs.append(event)
        print(f"EVENT: {event_type} - {details}")

    def export_logs(self, output_path: str = "output.json"):
        with open(output_path, 'w') as f:
            json.dump(self.event_logs, f, indent=4)
        print(f"Logs exported to {output_path}")

    def add_load(self, load: Load):
        self.plant.loads[load.load_id] = load
        self.pending_loads.append(load.load_id)
        print(f"Load {load.load_id} enqueued.")
        self.executor.submit(self.process_load, load.load_id)

    def process_load(self, load_id: str):
        if load_id in self.pending_loads:
            self.pending_loads.remove(load_id)
        
        load = self.plant.loads.get(load_id)
        if not load: return

        while load.current_process:
            process = load.current_process
            self.active_processes[load_id] = process.name
            
            # 1. Selection
            # We assume for simplicity the load starts in some row or needs to be loaded
            current_row_id = self._get_load_row(load) or "Row1" 
            target = self.scheduler.find_target_station(load, current_row_id)
            
            if not target:
                self.record_event("STATION_SEARCH_WAIT", {"load_id": load_id, "process": process.name})
                time.sleep(2)
                continue

            target_row_id, target_stn_id = target
            
            # 2. Row Lock & Move
            with self._get_row_lock(target_row_id):
                if target_row_id != current_row_id:
                    self._execute_cross_row_transfer(load, current_row_id, target_row_id)
                
                self._execute_move(load, target_row_id, target_stn_id)

            # 3. Process Execution
            self.record_event("PROCESS_START", {"load_id": load_id, "process": process.name, "duration": process.duration})
            time.sleep(process.duration / 10) # Speed up simulation (using duration/10 consistently if desired, or keep as is)
            # Original code used time.sleep(process.duration), but for CSV simulation which has long durations, 
            # maybe it's better to stay consistent. I'll use /10 for simulation speed.
            load.complete_current_process()
            self.record_event("PROCESS_COMPLETE", {"load_id": load_id, "process": process.name})
            self.save_state()

        print(f"Load {load_id} completed all processes.")
        self.active_processes.pop(load_id, None)
        self.record_event("LOAD_COMPLETE", {"load_id": load_id})

    def _get_row_lock(self, row_id: str) -> threading.Lock:
        if row_id not in self.locks:
            self.locks[row_id] = threading.Lock()
        return self.locks[row_id]

    def _get_load_row(self, load: Load) -> Optional[str]:
        # Helper to find which row the load is currently in
        for row_id, row in self.plant.rows.items():
            if any(s.station_id == load.current_station_id for s in row.stations):
                return row_id
        return None

    def _execute_move(self, load: Load, row_id: str, stn_id: str):
        row = self.plant.rows[row_id]
        target_stn = row.get_station_by_id(stn_id)
        transporter = row.transporter
        
        # Calculate time
        travel_time = self.scheduler.calculate_traversal_time(transporter, target_stn.position_mm)
        
        # Record Event
        self.record_event("MOVE_START", {
            "load_id": load.load_id,
            "transporter_id": transporter.transporter_id,
            "from_pos": transporter.position_mm,
            "to_stn": stn_id,
            "to_pos": target_stn.position_mm,
            "est_duration": travel_time
        })

        # Simulate physical movement
        time.sleep(travel_time / 10) # Speed up simulation for demo
        
        # Update state
        if load.current_station_id:
            old_stn = self._find_station(load.current_station_id)
            if old_stn: old_stn.current_load_id = None
            
        load.current_station_id = stn_id
        target_stn.current_load_id = load.load_id
        transporter.position_mm = target_stn.position_mm

        self.record_event("MOVE_COMPLETE", {"load_id": load.load_id, "station": stn_id})

    def _execute_cross_row_transfer(self, load, from_row, to_row):
        route = self.scheduler.get_transfer_route(load, from_row, to_row)
        self.record_event("TRANSFER_START", {"load_id": load.load_id, "from": from_row, "to": to_row})
        for step in route:
            # Simulate each step of the transfer
            time.sleep(0.1)
            self.record_event("TRANSFER_STEP", {"load_id": load.load_id, "step": step['type']})
        self.record_event("TRANSFER_COMPLETE", {"load_id": load.load_id})

    def _find_station(self, stn_id: str) -> Optional[Station]:
        for row in self.plant.rows.values():
            stn = row.get_station_by_id(stn_id)
            if stn: return stn
        return None

    def get_status(self):
        return {
            "active_loads": list(self.active_processes.keys()),
            "pending_queue": self.pending_loads,
            "plant_summary": {rid: [s.current_load_id for s in r.stations] for rid, r in self.plant.rows.items()}
        }
