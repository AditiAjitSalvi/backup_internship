from ..services.plant_manager import PlantManager
from ..domain.models import Load, Process, Row, Station, StationType, Transporter

class ROC_API:
    def __init__(self, manager: PlantManager):
        self.manager = manager

    def submit_load(self, processes_config: list):
        """
        Submits a load with a list of processes.
        processes_config: [{"name": "P1", "duration": 10}, ...]
        """
        processes = [Process(name=p["name"], duration=p["duration"]) for p in processes_config]
        load = Load(processes=processes)
        self.manager.add_load(load)
        return load.load_id

    def get_plant_status(self):
        return self.manager.get_status()

    def update_row_config(self, row: Row):
        self.manager.plant.rows[row.row_id] = row
        print(f"Row {row.row_id} configured.")

    def check_can_accept_load(self):
        # Basic check: is there at least one loading station available?
        for row in self.manager.plant.rows.values():
            for s in row.stations:
                if s.station_type == StationType.LOADING and s.is_available():
                    return True
        return False
