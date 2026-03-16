# Multitank Handling Logic in vADVANCED13.py

This document explains the core logic behind the multi-tank and multi-load scheduling system in `vADVANCED13.py`. It is structured to help you demonstrate how wagons are sequenced, how parts are routed through multiple tanks, and how collisions are prevented.

## 1. System Overview
The system is designed to fully automate the dispatching of multiple wagons across a yard containing various chemical/processing tanks. The script acts as a **Dynamic Discrete Event Scheduler** that processes multiple parallel loads. It evaluates physical constraints (travel distance, lift/lower time, wagon speed) and ensures no two wagons collide while trying to access the same station.

## 2. Tank Routing and Processing Modes
The system supports three distinct processing modes to route items through the plant. These modes determine how a job moves from one tank to another when multiple identical tanks are available.

### A. Grouped Processing (Pipeline Routing)
- **Enabled via:** `gap_analysis_grouped_sequence()`
- **Logic:** This is a data-driven pipeline generator. Instead of hardcoding paths, it reads the tank definitions from the CSV and groups consecutive tanks having the same `process_name`.
- **The Pipeline Pattern:** It naturally handles fan-out and fan-in scenarios. 
  - **Group 0** (e.g., Tank 1 - Loading) $\rightarrow$ One source.
  - **Group 1** (e.g., Tanks 2, 3, 4 - Parallel Acid Tanks) $\rightarrow$ Parts distribute to the earliest available empty tank.
  - **Group 2** (e.g., Tank 5 - Post Rinse) $\rightarrow$ Collector tank where all Group 1 tanks funnel into.
- **Why it's smart:** If an item is ready to move to Group 1, it checks Tanks 2, 3, and 4 and routes the item to whichever tank is open.

### B. Load Balance / Rotation Mode
- **Enabled via:** `filter_tanks_by_load_rotation()`
- **Logic:** Also groups consecutive tanks with the same `process_name` into "logical steps." However, it explicitly pins a load to a specific tank based on the load's index.
- **Mechanism:** `selected_tank = step_options[load_index % len(step_options)]`
- **Use Case:** If there are 4 zinc tanks, Load 0 goes to Tank 1, Load 1 goes to Tank 2, Load 2 to Tank 3, Load 3 to Tank 4, Load 4 back to Tank 1. This guarantees an even spread of work across all available identical tanks.

### C. Sequential Processing with Criticality
- **Enabled via:** `gap_analysis_sequence(processing_mode="sequential")` with `filter_tanks_sequential()`
- **Logic:** Every load processes the required tanks in exact order (e.g., Tank 1 $\rightarrow$ Tank 2 $\rightarrow$ Tank 3). Multiple loads can still run independently in parallel pipelines.
- **Criticality Checking:** Handles strict process timing.
  - **`HIGH` Criticality:** The wagon must pick up the item *immediately* once its dip time finishes.
  - **`LOW` Criticality:** If the wagon is busy, the item can wait in the tank up to a threshold (`MAX_HOLD_TIME_LOW = 300` seconds / 5 mins) before forced pickup.

## 3. The Scheduler Engine
Once the route for the load is decided, the core engine starts scheduling moves:
1. **Clock Tracking:** It maintains the current state of every wagon (`wagon_pos_mm`, `wagon_times`) and every station (`station_free_at`).
2. **Greedy Decision Making:** For every active load, it looks at the next required jump (`s_from` $\rightarrow$ `s_to`).
3. **Time Calculation:** It computes the travel time using a physics formula: `(Superfast / Fast / Slow speed zones)` + `lift time` + `lower time`.
4. **Action Selection:** Out of all possible loads waiting to move, the scheduler picks the one that can start the earliest (`min_start_time`).

## 4. Anti-Collision and Interlocks (Safety Engine)
With multiple wagons moving simultaneously, they will inevitably compete for the same physical space. The `CollisionEngine` prevents disasters.

### Station Occupancy Verification
- **Core Formula:** `actual_entry_time = max(wagon_arrival_time, station_free_time)`
- If `wagon_arrival_time` is earlier than when the station is free, a **wait** is required.
- **Interlock Strategy:** 
  - If the wait is **> 5 seconds**: The system injects a precise wait duration $\rightarrow$ `SET_INTERLOCK(wait_time_in_seconds)`.
  - If the wait is **$\le$ 5 seconds**: The system injects a dynamic sensor wait $\rightarrow$ `WAIT_INTERLOCK(STATION_n_FREE)`.

### Proximity Buffer (Secondary Safety)
- The system constantly checks the distance (`distance_mm`) between wagons operating on the same row.
- If two wagons are scheduled to get closer than `safe_distance_mm` (default: 600mm), the engine inserts a proximity warning command: `WAIT_INTERLOCK(PROXIMITY_BUFFER_wagonA_vs_wagonB)`. This stops the advancing wagon until the leading wagon vacates the clearance zone.

### Double-Check Correction Pass
- After the sequence is generated, a function called `run_collision_correction_pass()` reviews the entire timeline as a safety net. It rebuilds the occupancy table and injects any missed WAIT instructions to ensure 100% collision-free code is sent to the PLC.

---
**Demonstration Talking Points Summary:**
- **Dynamic Routing:** Show how items automatically find empty tanks (Grouped pipeline) vs assigned routing (Load balancing).
- **Physics Simulation:** Explain that timings are not guessed; they compute exact mm/sec distances + lift/lower times.
- **Safety First:** Highlight the Anti-collision logic—wagons never overlap, sharing stations triggers `WAIT_INTERLOCK` commands, and proximity zones keep wagons at a physical safe distance.
