
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

try:
    import vADVANCED13
    print("Import successful.")
    
    # Check default path
    print(f"Default tanks csv in module: {vADVANCED13.tanks_csv_default}")
    
    # Run function
    print("Calling generate_sequence_from_tanks()...")
    vADVANCED13.generate_sequence_from_tanks()
    print("Success!")
    
except Exception as e:
    import traceback
    traceback.print_exc()
