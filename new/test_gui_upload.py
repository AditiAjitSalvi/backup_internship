from streamlit.testing.v1 import AppTest
from unittest.mock import patch
import os
import shutil
from pathlib import Path

def test_upload():
    model_path = Path("model")
    wagon_config = model_path / "wagon_config.csv"
    tanks_csv = model_path / "tanks_csv.csv"
    
    # Backup existing
    if wagon_config.exists():
        shutil.copy(wagon_config, "model/wagon_config_backup.csv")
    if tanks_csv.exists():
        shutil.copy(tanks_csv, "model/tanks_csv_backup.csv")
        
    try:
        at = AppTest.from_file("v13_gui.py").run()
        print("Initial run successful")
        
        # Test wagon upload
        wagon_content = b"Transporter Name,Superfast Speed,Fast Speed,Slow Speed,Lift Time,Lower Time,Minimum Station No,Maximum Station No,Basic Position,Wagon Length (mm),Safety Buffer (mm),Size Factor,Station Occupy Factor,Row\nTest Wagon,1,2,3,4,5,1,10,0,1000,100,1,1,1\n"
        
        # There should be file uploaders. We can find them by label.
        # But our labels are collapsed, however they still have labels "Upload Wagon CSV" and "Upload Station CSV"
        # at.file_uploader(label="Upload Wagon CSV").set_value(...) doesn't work exact like that without checking labels
        wagon_upload = [fu for fu in at.file_uploader if fu.label == "Upload Wagon CSV"]
        if wagon_upload:
            wagon_upload[0].set_value(wagon_content).run()
            print("Wagon upload run done")
            with open(wagon_config, "r") as f:
                c = f.read()
            if "Test Wagon" in c:
                print("SUCCESS: Wagon config saved automatically via upload.")
            else:
                print("FAILURE: Wagon config NOT saved via upload.")
                
        # Test tanks upload
        tanks_content = b"project_id,station_no,process_name,critical_status,distance_mm,dip_time_sec,Row\n99,1,Upload Test,Low,100,10,1\n"
        tanks_upload = [fu for fu in at.file_uploader if fu.label == "Upload Station CSV"]
        if tanks_upload:
            tanks_upload[0].set_value(tanks_content).run()
            print("Tanks upload run done")
            with open(tanks_csv, "r") as f:
                c = f.read()
            if "Upload Test" in c:
                print("SUCCESS: Tanks config saved automatically via upload.")
            else:
                print("FAILURE: Tanks config NOT saved via upload.")

    finally:
        # Restore backups
        if os.path.exists("model/wagon_config_backup.csv"):
            shutil.copy("model/wagon_config_backup.csv", wagon_config)
            os.remove("model/wagon_config_backup.csv")
        if os.path.exists("model/tanks_csv_backup.csv"):
            shutil.copy("model/tanks_csv_backup.csv", tanks_csv)
            os.remove("model/tanks_csv_backup.csv")

if __name__ == "__main__":
    test_upload()
