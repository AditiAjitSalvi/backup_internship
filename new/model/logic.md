# Logic Documentation — `vADVANCED13.py`

## Overview

This file is the core **sequence generation engine** for a multi-wagon electroplating / dip-processing plant. It combines:

1. **Physics-based sequence generation** (travel time, dip time, lift/lower time)
2. **Anti-collision engine** (station occupancy tracking + WAIT insertion)
3. **AI / Reinforcement Learning layer** (PPO agent + Transformer + GNN) for optional AI-guided scheduling

---

## 1. Configuration & Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `LR` | 3e-4 | Adam optimizer learning rate |
| `GAMMA` | 0.99 | Discount factor for PPO returns |
| `EPS_CLIP` | 0.2 | PPO clipping coefficient |
| `K_EPOCHS` | 4 | PPO policy update epochs |
| `D_MODEL` | 64 | Transformer embedding dimension |
| `N_HEADS` | 4 | Attention heads |
| `N_LAYERS` | 2 | Transformer encoder layers |
| `MAX_STATIONS` | 200 | Maximum stations in the plant |
| `MAX_STEPS` | 50 | Max steps per episode |

The device is auto-selected: **CUDA if available, otherwise CPU**.

CSV file paths are auto-detected relative to the script's directory:
- `wagon_config.csv` — per-wagon speed, range, timing parameters
- `tanks_csv.csv` — station data (station number, distance, dip time)
- `row_config.csv` — per-row station range metadata

---

## 2. Data Loading

### `load_wagon_config()`
Reads `wagon_config.csv`. Each wagon's entry stores:
- `sf`, `f`, `s` — superfast / fast / slow speeds (m/min)
- `lift_time`, `lower_time` — vertical movement durations (sec)
- `min_stn`, `max_stn` — station range this wagon serves
- `basic_pos` — home/base station number
- `row` — which physical row the wagon is on
- `occupy_factor` — multiplier applied to completion time

### `load_row_config()`
Reads `row_config.csv` to provide per-row min/max station boundaries.

### `load_data()`
Reads historical step sequences from `sequnce for traing.csv` and cross-row adjacency from `CrossTrolleyMaster.csv`.  
Returns:
- `train_data` — list of `(cmd_idx, station_no)` sequences
- `adj_list` — graph adjacency for the Safety GNN
- `vocab_cmd` — command token vocabulary

### `verify_stations()`
Checks that the number of unique stations in tank data matches the wagon's configured `max_stn`. Prints pass/fail.

---

## 3. ML Models

### `SeqTransformer` (Transformer Encoder)
- Embeds `(command, station, position)` → summed embedding
- Passes through a standard PyTorch `TransformerEncoder`
- Outputs logits over commands and stations (for next-step prediction)

### `SafetyGNN` (Graph Safety Network)
- Embeds binary station occupancy states (0 = free, 1 = occupied)
- Performs one-hop neighborhood aggregation: `h_next = ReLU(W · A·h)`
- `A` = station adjacency matrix built from `CrossTrolleyMaster.csv`
- Outputs a per-station safety embedding

### `ActorCritic`
- Combines `SeqTransformer` context (last token) + `SafetyGNN` max-pooled embedding
- Concatenated state → three heads:
  - **Actor-cmd**: softmax over command vocabulary
  - **Actor-stn**: softmax over stations
  - **Critic**: scalar value estimate

### `PPOBuffer`
Simple replay buffer storing: command/station sequences, station states, sampled actions, log-probabilities, rewards, and terminal flags.

### `PPOAgent`
- Maintains `policy` (updated) and `policy_old` (reference for ratio calculation)
- `select_action()`: pads sequence to `MAX_STEPS`, queries `policy_old`, samples `(cmd, stn)` from Categorical distributions, stores in buffer
- `update()`: Monte Carlo return calculation → reward normalisation → K-epoch PPO clipped surrogate update → copies weights to `policy_old`

### `WagonEnv`
Simulated RL environment:
- State: binary station occupancy vector
- Reward: `+1` per step, `-50` collision (occupied station), `-20` adjacency conflict
- Episode ends at `MAX_STEPS`

---

## 4. Anti-Collision Engine

### Core Formula — `compute_entry_time()`
> **Single source of truth** used by all collision checks.

```
actual_entry_time = max(wagon_arrival_time, station_free_time)
wait_time         = actual_entry_time - wagon_arrival_time
```

| Condition | Instruction Generated |
|-----------|----------------------|
| `wait_time > 5 s` | `SET_INTERLOCK(wait_time)` — exact timed hold |
| `0 < wait_time ≤ 5 s` | `WAIT_INTERLOCK(STATION_n_FREE)` — poll PLC signal |
| `wait_time == 0` | No wait — enter immediately |

A wait instruction is **only inserted for cross-wagon conflicts** (a different wagon holds the station). A wagon waiting for its own dip task to finish advances the clock silently.

---

### `StationOccupancyTracker`
Central shared state for all wagons:

```
station_status[station_no] = {
    "free_at":     float,  # seconds
    "occupied_by": wagon_id or None
}
```

Key methods:
- `reserve(stn, wagon_id, entry_time, process_time)` → sets `free_at = entry_time + process_time`
- `release(stn)` → clears `occupied_by`
- `is_collision(stn, arrival)` → `arrival < free_at`

Only **shared stations** (used by ≥ 2 wagons) participate in collision tracking.

---

### `CollisionEngine`
Wraps `StationOccupancyTracker` and adds proximity checking.

**`compute_entry(wagon_id, station_no, wagon_arrival_time, process_time, distance_mm)`**  
Returns a dict:
- `actual_entry_time` — adjusted start time
- `wait_time` — delay imposed
- `wait_instructions` — list of `WAIT_INTERLOCK` / `SET_INTERLOCK` strings
- `collision_detected` — True if a cross-wagon conflict was found
- `proximity_warning` — True if another wagon is within `safe_distance_mm` (default 600 mm)

Proximity instructions are prepended before station-conflict instructions.

---

### `run_collision_correction_pass()`
**Second-pass safety net** over the already-built sequence table.

Algorithm:
1. Discover shared stations from the generated sequence
2. Pre-initialise `StationOccupancyTracker`
3. Walk rows in order; skip existing WAIT rows
4. For each station-entry row: apply `compute_entry_time()` formula
5. Insert `WAIT_INTERLOCK` + `SET_INTERLOCK` rows if a cross-wagon conflict is found
6. Reserve station; advance per-wagon clock

This catches any timing edge-cases missed by the real-time engine.

---

## 5. Physics-Based Sequence Generation

### `calculate_time_value()`
Converts distances to travel time using a 3-zone speed profile:

```
time = (d1 / superfast_mm_per_s) + (d2 / fast_mm_per_s) + (d3 / slow_mm_per_s)
```

Speed unit conversion: `speed_m_min × 16.66 = mm/s`  
`d3 = 500 mm` (fixed slow-approach zone)

### `build_wagon_routes()`
Partitions the flat `tanks` list into per-wagon ordered station lists.  
Assignment rule: station belongs to the wagon whose `row` and `[min_stn, max_stn]` range contains that station number.

### `identify_shared_stations()`
Returns the set of station numbers that appear in the routes of **two or more wagons** — these are the only stations that trigger collision-wait logic.

---

### `gap_analysis_sequence()` — Main Sequence Builder

Generates the unified multi-wagon step table:

**Headers (collision mode):** `Row, Wagon, Step No, Command, Value, TravelTime, AccumulatedTime, CollisionFlag, WaitTime`

**Step-by-step logic for each consecutive tank pair `(curr, next)`:**

1. **Identify wagon** — match station to wagon by row + range
2. **Wagon transition detection** — if wagon changes, close the previous wagon's block with a `GET FROM` row and release that station
3. **Full-cycle startup wait** — on a wagon's very first step, check if the previous wagon in logical order has finished; insert `WAIT_INTERLOCK + SET_INTERLOCK` if needed
4. **Anti-collision check for `s_curr`** — call `engine.compute_entry()` for the current station; insert any required WAIT rows; advance `acc_time` to `actual_entry_time`
5. **Pre-GET-FROM wait** — if `s_curr` is shared and still occupied by another wagon, insert `WAIT_INTERLOCK` + `SET_INTERLOCK`
6. **`GET FROM s_curr`** — lift action; add `lift_time` to accumulated time; release station signal
7. **Anti-collision check for `s_next`** — pre-reserve destination before travel; insert WAIT rows if needed
8. **Station reservation for `s_next`** — set `station_signal`, `station_occupied_by`, and `station_free_time`
9. **`PUT ON s_next`** — travel + lower action; add `(travel_time + lower_time)` to accumulated time
10. **Completion time tracking** — if this was the wagon's last hop, calculate full-cycle finish time (including return-to-base) for the next wagon's startup check

After all pairs are processed:
- Append final `GET FROM` for the last wagon
- Run `run_collision_correction_pass()` as a second safety pass
- Print collision engine summary

---

## 6. Public API Functions

### `generate_sequence_from_data(tanks_data, config, verbose)`
Entry point for GUI integration. Accepts a **list of dicts or pandas DataFrame**.  
Calls `gap_analysis_sequence()` and optionally prints the result as a formatted table.

### `generate_sequence_from_tanks(csv_path, tanks_data, config)`
Wrapper supporting both file-path and direct-data inputs (backward compatible).  
If `tanks_data` is provided, delegates to `generate_sequence_from_data()`.  
Otherwise, reads the CSV and calls `generate_sequence_from_data()`.

### `generate_sequence_ai(csv_path, model_path, config)`
Inference mode using the trained PPO model.  
Runs the `WagonEnv` simulation loop; outputs step table with AI-selected command + station pairs.  
(Physics timing not integrated in AI mode — travel times are recorded as `0.00`.)

---

## 7. Main Entry Point

`main()` parses CLI arguments and dispatches to one of three modes:

| `--mode` | Action |
|----------|--------|
| `gen` | Physics-based sequence generation from CSV |
| `ai_gen` | AI-guided inference using trained PPO model |
| `train` | RL training loop (1000 episodes, saves `model_v7.pth`) |

---

## 8. Data Flow Diagram

```
tanks_csv.csv ──────────────────────────────┐
wagon_config.csv ────────────────────────┐  │
                                         ▼  ▼
                               build_wagon_routes()
                               identify_shared_stations()
                                         │
                                         ▼
                           gap_analysis_sequence()
                           ┌─────────────────────────────┐
                           │  For each (curr → next) pair │
                           │  1. Assign wagon             │
                           │  2. Startup wait check       │
                           │  3. engine.compute_entry()   │
                           │     (WAIT rows inserted)     │
                           │  4. GET FROM curr            │
                           │  5. engine.compute_entry()   │
                           │     (WAIT rows for s_next)   │
                           │  6. PUT ON next              │
                           └─────────────────────────────┘
                                         │
                                         ▼
                      run_collision_correction_pass()  ← 2nd safety pass
                                         │
                                         ▼
                              sequence_data (list of lists)
                              [Row, Wagon, Step, Command, Value,
                               TravelTime, AccTime, CollisionFlag, WaitTime]
```

---

## 9. Key Design Decisions

- **Single formula rule**: `compute_entry_time()` is the one place the anti-collision formula lives; both the real-time engine and the correction pass call it.
- **Shared-stations only**: Collision tracking applies **only to stations used by ≥ 2 wagons**, avoiding false waits for single-wagon stations.
- **Cross-wagon only**: A WAIT row is only inserted when the blocking wagon is **different** from the requesting wagon.
- **Two-pass safety**: Real-time insertion during generation + post-generation correction pass = belt-and-suspenders approach.
- **Occupy factor**: Each wagon's completion time is scaled by `occupy_factor` to provide a configurable safety margin for startup sequencing.
- **Proximity buffer**: Secondary safety check (600 mm default) prevents wagons from physically crowding each other, independent of station occupancy.
