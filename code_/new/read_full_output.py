
try:
    with open("full_output.txt", "r", encoding="utf-16-le") as f:
        print(f.read())
except Exception as e:
    print(f"Error reading file: {e}")
