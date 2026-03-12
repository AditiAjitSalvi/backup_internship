"""
Removes the dead code block that accumulated between the new
gap_analysis_grouped_sequence return and the generate_sequence_from_data def.
"""
import re

with open("vADVANCED13.py", "r", encoding="utf-8") as f:
    content = f.read()

# The new clean function ends correctly with this exact snippet.
# Everything between the first occurrence of this snippet and
# 'def generate_sequence_from_data(' is dead code and should be deleted.

END_MARKER = "    rows.sort(key=lambda x: float(x[headers.index(\"AccumulatedTime\")]))\n    return [headers] + rows\n"
NEXT_DEF   = "\ndef generate_sequence_from_data("

# Find first and last positions
first_end = content.find(END_MARKER)
if first_end == -1:
    print("ERROR: END_MARKER not found")
    exit(1)

# Find where the next top-level function starts (after the dead code)
next_def_pos = content.find(NEXT_DEF, first_end + len(END_MARKER))
if next_def_pos == -1:
    print("ERROR: NEXT_DEF not found after END_MARKER")
    exit(1)

# Cut: keep everything up to (and including) the first clean return,
# then jump straight to the next function.
new_content = content[:first_end + len(END_MARKER)] + "\n" + content[next_def_pos + 1:]

with open("vADVANCED13.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Done. Removed {next_def_pos - (first_end + len(END_MARKER))} chars of dead code.")
