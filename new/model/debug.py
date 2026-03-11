# standalone logic reproduction

tanks = [
    {"station_no": "1", "process_name": "Loading", "dip_time_sec": "0"},
    {"station_no": "2", "process_name": "A acid", "dip_time_sec": "10"},
    {"station_no": "3", "process_name": "A acid", "dip_time_sec": "10"},
    {"station_no": "4", "process_name": "Rinse", "dip_time_sec": "5"}
]

def execute_move(c, n, block_idx=None, job_idx=None):
    print(f"Move from {c['station_no']} to {n['station_no']} (block={block_idx}, job={job_idx})")
    return "W1"

def group_same_process_blocks(tank_list):
    if not tank_list: return []
    blocks = []
    current_block = [tank_list[0]]
    for i in range(1, len(tank_list)):
        curr = tank_list[i]
        prev = current_block[-1]
        p1 = prev.get('process_name', '').strip().lower()
        p2 = curr.get('process_name', '').strip().lower()
        if p1 == p2 and p1 and p1 not in ('end', 'loading', ''):
            current_block.append(curr)
        else:
            blocks.append(current_block)
            current_block = [curr]
    if current_block:
        blocks.append(current_block)
    return blocks

print("Grouping...")
blocks = group_same_process_blocks(tanks)
for i, b in enumerate(blocks):
    print(f"Block {i}: {[t['station_no'] for t in b]}")

has_parallel = any(len(b) > 1 for b in blocks)
print(f"Has parallel: {has_parallel}")

accumulated_times = {"W1": 0.0}

active_jobs = []

for b_idx in range(len(blocks) - 1):
    curr_block = blocks[b_idx]
    next_block = blocks[b_idx + 1]
    
    print(f"--- Processing Block {b_idx} -> {b_idx+1} ---")
    
    if len(curr_block) == 1 and not active_jobs:
        print("Sequential move")
        execute_move(curr_block[0], next_block[0])
    else:
        print(f"Parallel block processing with {len(active_jobs)} existing jobs")
        if not active_jobs:
            for j_idx, curr_tank in enumerate(curr_block):
                n_tank = next_block[j_idx % len(next_block)]
                w_id = execute_move(curr_tank, n_tank, block_idx=b_idx, job_idx=j_idx)
                if w_id:
                    dip = float(n_tank.get('dip_time_sec', 0))
                    active_jobs.append({
                        "wagon_id": w_id,
                        "tank_obj": n_tank,
                        "free_at": accumulated_times[w_id] + dip,
                        "block_idx": b_idx + 1,
                        "job_idx": j_idx
                    })
        
        counter = 0
        while active_jobs and counter < 10:
            counter += 1
            active_jobs.sort(key=lambda x: x["free_at"])
            job = active_jobs.pop(0)

            w_id = job["wagon_id"]
            j_free = job["free_at"]
            j_b_idx = job["block_idx"]
            j_idx = job["job_idx"]
            
            print(f"  Got Job {j_idx}: w={w_id}, b_idx={j_b_idx}, free={j_free}")

            if accumulated_times[w_id] < j_free:
                accumulated_times[w_id] = j_free

            if j_b_idx < len(blocks) - 1:
                c_tank = job["tank_obj"]
                n_block = blocks[j_b_idx + 1]
                n_tank = n_block[j_idx % len(n_block)]
                
                w_id = execute_move(c_tank, n_tank, block_idx=j_b_idx, job_idx=j_idx)
                if w_id and j_b_idx + 1 < len(blocks) - 1:
                    dip = float(n_tank.get('dip_time_sec', 0))
                    active_jobs.append({
                        "wagon_id": w_id,
                        "tank_obj": n_tank,
                        "free_at": accumulated_times[w_id] + dip,
                        "block_idx": j_b_idx + 1,
                        "job_idx": j_idx
                    })
            else:
                print(f"  Job {j_idx} reached end of blocks")
        if counter == 10:
            print("  [ERROR] Loop limit reached!")

print("Done")
