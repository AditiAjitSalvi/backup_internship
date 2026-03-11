# Sequential Processing Implementation Plan

## Overview

Replace the current load-based linear logic with a sequential and flexible processing model that:
- Distributes items sequentially across tanks
- Allows independent completion and movement of items
- Handles criticality-based priority
- Supports flexible movement rather than strictly linear flow

---

## Current Logic Analysis

### Existing Implementation (`filter_tanks_by_load_rotation`)
- Groups consecutive tanks with the same `process_name` into logical steps
- Selects one tank per step using `load_index % len(step_options)`
- Creates load balancing but NOT sequential processing

### Scheduling (`gap_analysis_sequence`)
- Event-driven, processes based on timing
- Uses load-index based selection

---

## Implementation Phases

### Phase 1: Add Criticality Support

1. **Create `get_tank_criticality()` function**
   - Check for `criticality` column in tank data
   - Default to "HIGH" if not specified
   - Values: "HIGH" or "LOW"

2. **Add configuration parameter**
   - `max_hold_time`: Maximum time LOW criticality items can wait (default: 300 seconds = 5 minutes)

---

### Phase 2: New Sequential Tank Selection

Create `filter_tanks_sequential()` function:

```python
def filter_tanks_sequential(tanks, load_index=0):
    """
    Sequential processing: Tank 1 → Tank 2 → Tank 3
    Each load processes ALL tanks in order (not modulo-based selection)
    Multiple loads run INDEPENDENTLY in parallel
    """
```

**Key Difference from Current Logic:**

| Aspect | Current (Load Balance) | Proposed (Sequential) |
|--------|----------------------|----------------------|
| Tank Selection | Modulo-based per step | Sequential 1→2→3 |
| Load 1 Path | Step1:Tank1, Step2:Tank1... | Tank1→Tank2→Tank3 |
| Load 2 Path | Step1:Tank2, Step2:Tank2... | Tank1→Tank2→Tank3 |
| Loads | Interleaved/Balanced | Independent/Parallel |

---

### Phase 3: Modify `gap_analysis_sequence()`

#### New State Tracking

Add per-load state:
```python
load_states[l_idx] = {
    'next_hop': 0,
    'ready_at': 0.0,                    # When tank is ready for pickup
    'tank_complete_at': 0.0,           # When processing completes (arrival + dip_time)
    'current_stn': station_no,
    'criticality': "HIGH"/"LOW",
    'waiting_since': None,              # For LOW criticality tracking
    'finished': False
}
```

#### Criticality-Aware Scheduling Logic

**HIGH Criticality:**
- When `current_time >= tank_complete_at`, schedule pickup IMMEDIATELY
- No waiting for wagon availability preferred, but wait if necessary

**LOW Criticality:**
- When `current_time >= tank_complete_at`, wagon can wait up to 5 minutes (300s)
- If `current_time >= tank_complete_at + 300`, force pickup on next available wagon

---

### Phase 4: Flexible Movement Logic

#### Current Behavior (to replace)
- Items move based on load conditions
- Linear pipeline flow

#### New Behavior
- Each load tracks its own processing completion time independently
- Wagons can pick up from tanks as soon as:
  1. Tank processing completes (dip_time_sec elapsed)
  2. Wagon is available at that station
- No fixed load condition required

#### Movement Rules

1. **Tank Processing Completion**
   ```
   tank_complete_at = arrival_time + dip_time_sec
   ```

2. **Pickup Scheduling**
   - HIGH criticality: Pickup when `current_time >= tank_complete_at`
   - LOW criticality: Pickup when `current_time >= tank_complete_at` BUT can wait up to 5 min

3. **Wagon Assignment**
   - Each load independently schedules its pickups
   - No cross-load synchronization required
   - Collision engine handles shared station conflicts

---

## Key Changes Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `get_tank_criticality()` | NEW - Parse criticality field | Enable priority handling |
| `filter_tanks_sequential()` | NEW - Sequential selection | Replace modulo logic |
| `gap_analysis_sequence()` | MODIFY - Time-driven scheduling | Enable flexible movement |
| Load state tracking | ADD - tank_complete_at, waiting_since | Track per-load completion |
| Collision handling | RETAIN - Existing collision engine | Still prevents conflicts |

---

## Data Requirements

### Tank CSV Optional Columns

| Column | Values | Default | Description |
|--------|--------|---------|-------------|
| criticality | HIGH, LOW | HIGH | Item priority level |
| dip_time_sec | float | 0 | Processing time in tank |

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_HOLD_TIME_LOW` | 300 (seconds) | Max wait time for LOW criticality |
| `CRITICALITY_DEFAULT` | "HIGH" | Default if not specified |

---

## Example Behavior

### Scenario: 3 Tanks with Acid A

**Tank Configuration:**
- Tank 1: dip_time=60s, criticality=HIGH
- Tank 2: dip_time=90s, criticality=LOW
- Tank 3: dip_time=45s, criticality=HIGH

**Expected Flow (Single Load):**

1. Wagon goes to Tank 1
2. Tank 1 processes for 60s
3. HIGH criticality → Wagon picks up immediately at t=60s
4. Wagon moves to Tank 2
5. Tank 2 processes for 90s
6. LOW criticality → Wagon can wait up to 5 min
   - If wagon available at t=90s: Pickup immediately
   - If wagon busy: Wait until available (max 5 min)
   - If wagon unavailable after 5 min: Force pickup
7. Wagon moves to Tank 3
8. Tank 3 processes for 45s
9. HIGH criticality → Wagon picks up immediately at t=135s

**Multiple Loads (Independent):**

- Load 1: Tank1→Tank2→Tank3 (starts at t=0)
- Load 2: Tank1→Tank2→Tank3 (starts at t=60, offset)
- Both loads run in parallel, no cross-dependency

---

## Implementation Priority

1. **Phase 1 & 2**: Add criticality parsing and sequential selection
2. **Phase 3**: Modify scheduling logic
3. **Phase 4**: Test and refine flexible movement

---

## Backward Compatibility

- Keep existing `filter_tanks_by_load_rotation()` function
- New `filter_tanks_sequential()` as optional alternative
- Existing collision engine remains unchanged
- Add parameter to choose between load-balanced vs sequential logic
