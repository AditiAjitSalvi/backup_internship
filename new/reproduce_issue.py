
import sys
import os
import traceback

# Add the directory to sys.path so we can import the module
sys.path.append(os.getcwd())

try:
    import vADVANCED13
    print(f"Successfully imported vADVANCED13")
    print(f"Default tanks csv: {vADVANCED13.tanks_csv_default}")
    print(f"Default wagon config: {vADVANCED13.wagon_config_csv}")
    
    # Run the function that failed
    print("Running generate_sequence_from_tanks()...")
    vADVANCED13.generate_sequence_from_tanks()
    
except Exception:
    traceback.print_exc()
