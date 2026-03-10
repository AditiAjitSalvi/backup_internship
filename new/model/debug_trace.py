import sys

def trace_calls(frame, event, arg):
    if event == 'line':
        print(f"Executing: {frame.f_code.co_name} at line {frame.f_lineno}")
    return trace_calls

sys.settrace(trace_calls)

try:
    import test_loop
except Exception as e:
    print(f"Exception: {e}")
