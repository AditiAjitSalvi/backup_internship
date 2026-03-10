
try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"Torch location: {torch.__file__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
