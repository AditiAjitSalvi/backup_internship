import time
from scheduling_ROC.domain.models import Row, Station, StationType, Transporter, CrossTrolley
from scheduling_ROC.services.plant_manager import PlantManager
from scheduling_ROC.api.interface import ROC_API

def setup_sample_plant(api: ROC_API):
    # Configure Row 1
    row1 = Row(row_id="Row1", transporter=Transporter(transporter_id="T1", speed_mm_s=100))
    row1.stations = [
        Station(station_id="S1_LOAD", station_type=StationType.LOADING, position_mm=0),
        Station(station_id="S1_TANK1", station_type=StationType.TANK, position_mm=1000, compatible_processes=["Degreasing"]),
        Station(station_id="S1_IF", station_type=StationType.CROSS_TROLLEY_INTERFACE, position_mm=2000),
        Station(station_id="S1_UNLOAD", station_type=StationType.UNLOADING, position_mm=3000),
    ]
    api.update_row_config(row1)

    # Configure Row 2
    row2 = Row(row_id="Row2", transporter=Transporter(transporter_id="T2", speed_mm_s=120))
    row2.stations = [
        Station(station_id="S2_IF", station_type=StationType.CROSS_TROLLEY_INTERFACE, position_mm=0),
        Station(station_id="S2_TANK1", station_type=StationType.TANK, position_mm=1000, compatible_processes=["Plating"]),
        Station(station_id="S2_BUF", station_type=StationType.BUFFER, position_mm=2000),
        Station(station_id="S2_UNLOAD", station_type=StationType.UNLOADING, position_mm=3000),
    ]
    api.update_row_config(row2)

    # Configure Cross Trolley
    api.manager.plant.cross_trolleys.append(CrossTrolley(trolley_id="CT1", connected_rows=["Row1", "Row2"]))

if __name__ == "__main__":
    manager = PlantManager()
    api = ROC_API(manager)
    
    setup_sample_plant(api)

    if api.check_can_accept_load():
        print("System ready. Submitting load...")
        load_id = api.submit_load([
            {"name": "Degreasing", "duration": 2},
            {"name": "Plating", "duration": 3}
        ])
        print(f"Load {load_id} submitted.")

    # Let it run for a bit to see logs
    for _ in range(20):
        time.sleep(1)
        # print(api.get_plant_status())

    print("\nSimulation complete.")
