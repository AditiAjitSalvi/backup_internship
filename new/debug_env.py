import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("\nSearch Paths (sys.path):")
for path in sys.path:
    print(f"  - {path}")

print("\nChecking for 'tabulate'...")
try:
    import tabulate
    print(f"SUCCESS: 'tabulate' found at {os.path.dirname(tabulate.__file__)}")
    print(f"Version: {tabulate.__version__}")
except ImportError:
    print("FAILURE: 'tabulate' not found in this environment.")

print("\nChecking for 'torch'...")
try:
    import torch
    print(f"SUCCESS: 'torch' found at {os.path.dirname(torch.__file__)}")
except ImportError:
    print("FAILURE: 'torch' not found in this environment.")
