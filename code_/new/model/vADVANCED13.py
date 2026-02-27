import os
import csv
import random
import math
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tabulate import tabulate

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Auto-detect wagon config
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

# Specifically look for the nested path mentioned by the user
wagon_config_csv = os.path.join(base_dir, "wagon_config.csv")
tanks_csv_default = os.path.join(base_dir, "tanks_csv.csv")
row_config_csv = os.path.join(base_dir, "row_config.csv")

station_master_csv = r"e:\Internship\WayTime-dB.mdb\StationMaster.csv"
csv_programs = os.path.join(base_dir, "sequnce for traing.csv")
csv_zones = r"e:\Internship\WayTime-dB.mdb\CrossTrolleyMaster.csv"

# RL Hyperparameters
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
MAX_STATIONS = 200 
MAX_STEPS = 50     
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DATA LOADING & VERIFICATION
# ==========================================
def load_wagon_config(config_path=wagon_config_csv):
    """
    Loads wagon configuration from CSV.
    """
    config = {}
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                w_num = (row.get('Transporter Name') or row.get('Wagon Number') or "").strip()
                if not w_num:
                    continue
                
                config[w_num] = {
                    'sf': float(row.get('Superfast Speed', 0)),
                    'f': float(row.get('Fast Speed', 0)),
                    's': float(row.get('Slow Speed', 0)),
                    'lift_time': float(row.get('Lift Time', 0)),
                    'lower_time': float(row.get('Lower Time', 0)),
                    'min_stn': int(row.get('Minimum Station No', 1)),
                    'max_stn': int(row.get('Maximum Station No', 200)),
                    'basic_pos': int(row.get('Basic Position', 0)),
                    'row': int(row.get('Row') or row.get('Row Number') or 1),
                    'occupy_factor': float(row.get('Station Occupy Factor', 1.0))
                }
    except Exception as e:
        print(f"Error loading wagon config: {e}")
    return config

def load_row_config(config_path=row_config_csv):
    """ Loads row configuration from CSV. """
    rows = {}
    if not os.path.exists(config_path):
        return rows
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_num = int(row.get('Row Number', 1))
                rows[r_num] = {
                    'min_stn': int(row.get('First Station No', 1)),
                    'max_stn': int(row.get('Last Station No', 200))
                }
    except Exception as e:
        print(f"Error loading row config: {e}")
    return rows


def verify_stations(tanks, min_stn, max_stn):
    """
    Verifies if the total number of unique stations in the tank list matches 
    the expected count (Maximum Station No).
    """
    present_stations = {tank.get('station_no') for tank in tanks if tank.get('station_no')}
    total_count = len(present_stations)
    expected_count = max_stn

    print(f"\n[STATION VERIFICATION]")
    print(f"Goal: Total stations should be {expected_count}")
    print(f"Status: Found {total_count} stations in the tank data.")
    
    if total_count == expected_count:
        print(f"SUCCESS: Total station count matches Maximum Station No ({expected_count}).")
        return True
    else:
        diff = expected_count - total_count
        status = "missing" if diff > 0 else "extra"
        print(f"FAILURE: Total station count ({total_count}) does not match Maximum Station No ({expected_count}).")
        print(f"There are {abs(diff)} {status} stations.")
        return False

def load_data():
    """ Loads historical sequence data for training RL models. """
    print("Loading historical data for AI training...")
    sequences = defaultdict(list)
    vocab_cmd = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    
    if not os.path.exists(csv_programs):
        print(f"Warning: {csv_programs} not found. AI training will use empty data.")
        return [], {}, vocab_cmd

    with open(csv_programs, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # seq_id,project_id,wagon_no,step_no,command,station_no
                seq_id = row.get('seq_id') + "_" + row.get('project_id')
                cmd = row.get('command')
                stn = int(row.get('station_no')) if row.get('station_no') else 0
                if cmd not in vocab_cmd:
                    vocab_cmd[cmd] = len(vocab_cmd)
                sequences[seq_id].append((vocab_cmd[cmd], stn))
            except Exception as e:
                # print(f"Skipping row: {e}") 
                continue

    train_data = list(sequences.values())
    print(f"Loaded {len(train_data)} sequences for training.")
    
    adj_list = defaultdict(list)
    if os.path.exists(csv_zones):
        with open(csv_zones, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s1 = int(row['Row1StationNo'])
                    s2 = int(row['Row2StationNo'])
                    adj_list[s1].append(s2)
                    adj_list[s2].append(s1)
                except:
                    continue
                
    return train_data, adj_list, vocab_cmd

# ==========================================
# ML MODELS (TRANSFORMER + GNN + PPO)
# ==========================================
class SeqTransformer(nn.Module):
    def __init__(self, n_cmds, n_stations, d_model=D_MODEL):
        super(SeqTransformer, self).__init__()
        self.cmd_emb = nn.Embedding(n_cmds, d_model)
        self.stn_emb = nn.Embedding(n_stations, d_model)
        self.pos_emb = nn.Embedding(MAX_STEPS, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=N_HEADS, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.fc_cmd = nn.Linear(d_model, n_cmds)
        self.fc_stn = nn.Linear(d_model, n_stations)

    def forward(self, cmd_seq, stn_seq):
        seq_len = cmd_seq.size(1)
        pos = torch.arange(seq_len, device=cmd_seq.device).unsqueeze(0).expand(cmd_seq.size(0), -1)
        x = self.cmd_emb(cmd_seq) + self.stn_emb(stn_seq) + self.pos_emb(pos)
        x = self.transformer(x)
        return self.fc_cmd(x), self.fc_stn(x)

class SafetyGNN(nn.Module):
    def __init__(self, n_stations, d_model=D_MODEL):
        super(SafetyGNN, self).__init__()
        self.state_emb = nn.Embedding(2, d_model)
        self.w = nn.Linear(d_model, d_model)

    def forward(self, station_states, adj_matrix):
        x = self.state_emb(station_states)
        out = []
        for i in range(x.size(0)):
            h = x[i]
            aggr = torch.mm(adj_matrix, h) 
            h_next = torch.relu(self.w(aggr))
            out.append(h_next)
        return torch.stack(out)

class PPOBuffer:
    def __init__(self):
        self.cmd_seqs = []
        self.stn_seqs = []
        self.station_states = []
        self.actions_cmd = []
        self.actions_stn = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    def clear(self):
        del self.cmd_seqs[:]
        del self.stn_seqs[:]
        del self.station_states[:]
        del self.actions_cmd[:]
        del self.actions_stn[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, n_cmds, n_stations, d_model=D_MODEL):
        super(ActorCritic, self).__init__()
        self.transformer = SeqTransformer(n_cmds, n_stations, d_model)
        self.gnn = SafetyGNN(n_stations, d_model)
        self.actor_cmd = nn.Linear(d_model * 2, n_cmds)
        self.actor_stn = nn.Linear(d_model * 2, n_stations)
        self.critic = nn.Linear(d_model * 2, 1)

    def forward(self, cmd_seq, stn_seq, station_states, adj_matrix):
        # Extract context from Transformer
        seq_len = cmd_seq.size(1)
        pos = torch.arange(seq_len, device=cmd_seq.device).unsqueeze(0).expand(cmd_seq.size(0), -1)
        x_seq = self.transformer.cmd_emb(cmd_seq) + self.transformer.stn_emb(stn_seq) + self.transformer.pos_emb(pos)
        x_seq = self.transformer.transformer(x_seq)
        ctx_seq = x_seq[:, -1, :] 

        # Extract context from Safety GNN
        x_gnn = self.gnn(station_states, adj_matrix)
        ctx_safe, _ = torch.max(x_gnn, dim=1)
        
        state = torch.cat([ctx_seq, ctx_safe], dim=-1)
        return torch.softmax(self.actor_cmd(state), dim=-1), torch.softmax(self.actor_stn(state), dim=-1), self.critic(state)

class PPOAgent:
    def __init__(self, vocab_cmd, n_stations, adj_list):
        self.vocab_cmd = vocab_cmd
        self.n_stations = n_stations
        self.adj_matrix = self._build_adj_matrix(adj_list, n_stations)
        self.policy = ActorCritic(len(vocab_cmd), n_stations).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ActorCritic(len(vocab_cmd), n_stations).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = PPOBuffer()
        self.MseLoss = nn.MSELoss()
        
    def _build_adj_matrix(self, adj_list, n):
        mat = torch.zeros((n, n), device=device)
        for u, neighbors in adj_list.items():
            if u < n:
                for v in neighbors:
                    if v < n:
                        mat[u, v] = 1.0
        return mat

    def select_action(self, cmd_seq, stn_seq, station_states):
        # Use pre-padding so that the last element of the sequence is always the current state
        pad_len = MAX_STEPS - len(cmd_seq)
        padded_cmd = [self.vocab_cmd["<PAD>"]] * pad_len + cmd_seq
        padded_stn = [0] * pad_len + stn_seq
        
        # Prepare tensors for the old policy
        cmd_tensor = torch.tensor([padded_cmd], dtype=torch.long).to(device)
        stn_tensor = torch.tensor([padded_stn], dtype=torch.long).to(device)
        state_tensor = torch.tensor([station_states], dtype=torch.long).to(device)
        
        with torch.no_grad():
            probs_cmd, probs_stn, val = self.policy_old(cmd_tensor, stn_tensor, state_tensor, self.adj_matrix)
        
        dist_cmd = Categorical(probs_cmd)
        dist_stn = Categorical(probs_stn)
        
        action_cmd = dist_cmd.sample()
        action_stn = dist_stn.sample()
        
        # Store in buffer (use the actual index of the current step for context)
        # We need to take the probability corresponding to the last ACTUAL step
        last_idx = len(cmd_seq) - 1
        
        self.buffer.cmd_seqs.append(torch.tensor(padded_cmd, dtype=torch.long))
        self.buffer.stn_seqs.append(torch.tensor(padded_stn, dtype=torch.long))
        self.buffer.station_states.append(torch.tensor(station_states, dtype=torch.long))
        self.buffer.actions_cmd.append(action_cmd)
        self.buffer.actions_stn.append(action_stn)
        self.buffer.logprobs.append(dist_cmd.log_prob(action_cmd) + dist_stn.log_prob(action_stn))
        
        return action_cmd.item(), action_stn.item()

    def update(self):
        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_cmd_seqs = torch.stack(self.buffer.cmd_seqs).to(device)
        old_stn_seqs = torch.stack(self.buffer.stn_seqs).to(device)
        old_station_states = torch.stack(self.buffer.station_states).to(device)
        old_actions_cmd = torch.stack(self.buffer.actions_cmd).to(device)
        old_actions_stn = torch.stack(self.buffer.actions_stn).to(device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(device)

        # Optimize policy for K epochs:
        for _ in range(K_EPOCHS):
            # Evaluating old actions and values
            probs_cmd, probs_stn, state_values = self.policy(old_cmd_seqs, old_stn_seqs, old_station_states, self.adj_matrix)
            
            dist_cmd = Categorical(probs_cmd)
            dist_stn = Categorical(probs_stn)
            
            logprobs = dist_cmd.log_prob(old_actions_cmd) + dist_stn.log_prob(old_actions_stn)
            dist_entropy = dist_cmd.entropy() + dist_stn.entropy()
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()

class WagonEnv:
    def __init__(self, adj_list, max_stations):
        self.adj_list = adj_list
        self.max_stations = max_stations
        self.reset()
    def reset(self):
        self.station_occupancy = [0] * self.max_stations
        self.step_count = 0
        return self.station_occupancy
    def step(self, cmd, stn):
        reward = 0
        done = False
        self.step_count += 1
        if stn >= self.max_stations: return self.station_occupancy, -50, True
        if self.station_occupancy[stn] == 1: reward -= 50
        elif any(self.station_occupancy[n] == 1 for n in self.adj_list.get(stn, [])): reward -= 20
        self.station_occupancy[stn] = 1 
        reward += 1
        if self.step_count >= MAX_STEPS: done = True
        return self.station_occupancy, reward, done

def identify_shared_stations(tanks, config):
    """
    Identify stations that appear in 2 or more wagon sequences.
    """
    station_to_wagons = defaultdict(set)
    
    # Pre-build wagon routes to see which stations each wagon actually uses
    wagon_routes = build_wagon_routes(tanks, config)
    
    for wagon_id, route in wagon_routes.items():
        for tank in route:
            try:
                # Handle both dict and list (if coming from correction pass)
                if isinstance(tank, dict):
                    stn = int(tank.get('station_no', 0))
                elif isinstance(tank, (list, tuple)):
                    # This branch is for internal calls where route might be rows
                    continue 
                else:
                    continue
                station_to_wagons[stn].add(wagon_id)
            except (ValueError, TypeError, AttributeError):
                continue
                
    shared_stations = {stn for stn, wagons in station_to_wagons.items() if len(wagons) >= 2}
    return shared_stations

# ==========================================
# ANTI-COLLISION ENGINE
# ==========================================

class StationOccupancyTracker:
    """
    Tracks when each station becomes free and which wagon is currently occupying it.
    Used as the central state for collision detection across all wagon sequences.

    station_status structure:
        {
            station_no: {
                "free_at":     <float seconds>,   # when station becomes available again
                "occupied_by": <wagon_id or None>  # which wagon last reserved it
            }
        }
    """
    def __init__(self, shared_stations=None):
        self.station_status = {}
        self.shared_stations = set(shared_stations) if shared_stations is not None else set()

    # ------------------------------------------------------------------
    #  Initialisation helpers
    # ------------------------------------------------------------------
    def _ensure(self, station_no):
        """ Lazily initialise a station entry on first access. """
        if station_no not in self.station_status:
            self.station_status[station_no] = {"free_at": 0.0, "occupied_by": None}

    def initialize_from_stations(self, station_list):
        """
        Pre-populate every station in station_list so the occupancy table is
        fully visible from the start of scheduling (not lazily built on demand).

        Parameters
        ----------
        station_list : iterable of int
            All station numbers that exist in the plant.
        """
        for stn in station_list:
            try:
                self._ensure(int(stn))
            except (TypeError, ValueError):
                continue

    # ------------------------------------------------------------------
    #  Query helpers
    # ------------------------------------------------------------------
    def get_free_at(self, station_no):
        """ Return the time (seconds) at which station_no becomes free. """
        self._ensure(station_no)
        return self.station_status[station_no]["free_at"]

    def get_occupied_by(self, station_no):
        """ Return the wagon_id currently holding station_no (or None). """
        self._ensure(station_no)
        return self.station_status[station_no]["occupied_by"]

    # ------------------------------------------------------------------
    #  Mutation helpers
    # ------------------------------------------------------------------
    def reserve(self, station_no, wagon_id, entry_time, process_time):
        """
        Mark station_no as occupied by wagon_id from entry_time.
        Updates free_at = entry_time + process_time.
        (Only for shared stations as per system requirements)
        """
        if self.shared_stations and station_no not in self.shared_stations:
            return

        self._ensure(station_no)
        self.station_status[station_no]["free_at"]     = entry_time + process_time
        self.station_status[station_no]["occupied_by"] = wagon_id

    def release(self, station_no):
        """
        Explicitly mark station_no as unoccupied (occupied_by=None).
        Typically happens after a GET FROM command.
        """
        if self.shared_stations and station_no not in self.shared_stations:
            return
        self._ensure(station_no)
        self.station_status[station_no]["occupied_by"] = None

    # ------------------------------------------------------------------
    #  Collision check
    # ------------------------------------------------------------------
    def is_collision(self, station_no, wagon_arrival_time):
        """
        Returns True if wagon_arrival_time < station free_at,
        i.e. a collision would occur if the wagon entered right now.
        Only applies to shared stations.
        """
        if self.shared_stations and station_no not in self.shared_stations:
            return False
        return wagon_arrival_time < self.get_free_at(station_no)

    def summary(self):
        """ Return the full station status table for logging/debugging. """
        return dict(self.station_status)


# ==========================================
# CORE ANTI-COLLISION FORMULA (PUBLIC UTILITY)
# ==========================================

def compute_entry_time(wagon_arrival_time: float, station_free_time: float):
    """
    ⭐ The definitive anti-collision formula — single source of truth.

        actual_entry_time = max(wagon_arrival_time, station_free_time)
        wait_time         = actual_entry_time - wagon_arrival_time

    Used identically by CollisionEngine.compute_entry() and
    run_collision_correction_pass() so the core formula never diverges.

    Parameters
    ----------
    wagon_arrival_time : float  – when the wagon naturally reaches the station (s)
    station_free_time  : float  – earliest time the station is unoccupied (s)

    Returns
    -------
    (actual_entry_time, wait_time)  both float (seconds).
    wait_time > 0  means a WAIT instruction is required.

    Pattern recognition rule
    ------------------------
    if wait_time > WAIT_THRESHOLD (5 s)  →  WAIT_SEC(wait_time)   # exact duration known
    elif wait_time > 0                   →  WAIT_INT(STATION_n_FREE)  # poll PLC signal
    else                                 →  no wait (enter immediately)
    """
    actual_entry_time = max(wagon_arrival_time, station_free_time)
    wait_time         = actual_entry_time - wagon_arrival_time
    return actual_entry_time, wait_time


class CollisionEngine:
    """
    Anti-collision layer that sits on top of the physics-based sequence generator.
    Inserts WAIT_SEC or WAIT_INT instructions whenever two wagons compete for the
    same station, preventing overlapping occupancy.

    Rules
    -----
    * wait_time > 5 s  → WAIT_SEC(wait_time)
    * 0 < wait_time ≤ 5 s → WAIT_INT(STATION_<n>_FREE)
    * wait_time == 0       → no wait needed

    Movement Buffer (secondary safety)
    ------------------------------------
    If two wagons are physically closer than `safe_distance_mm` the later
    wagon also receives a WAIT_INT(PROXIMITY_BUFFER) instruction.
    """

    WAIT_THRESHOLD_SEC = 5          # seconds – boundary between WAIT_SEC and WAIT_INT
    DEFAULT_SAFE_DISTANCE_MM = 600  # mm – minimum gap between wagons

    def __init__(self, safe_distance_mm=None, shared_stations=None):
        self.tracker = StationOccupancyTracker(shared_stations=shared_stations)
        self.safe_distance_mm = safe_distance_mm or self.DEFAULT_SAFE_DISTANCE_MM
        # {wagon_id: current_position_mm}
        self.wagon_positions = {}

    # ------------------------------------------------------------------
    # Core collision check
    # ------------------------------------------------------------------
    def compute_entry(self, wagon_id, station_no, wagon_arrival_time, process_time,
                      distance_mm=None):
        """
        Determine the actual entry time, wait time, and any WAIT instructions
        required for wagon_id to enter station_no.

        Parameters
        ----------
        wagon_id        : str   – name of the wagon
        station_no      : int   – station being requested
        wagon_arrival_time : float – when the wagon would naturally arrive (seconds)
        process_time    : float – how long the wagon occupies the station (seconds)
        distance_mm     : float – current wagon position, for proximity buffer check

        Returns
        -------
        dict with keys:
            actual_entry_time   : float
            wait_time           : float
            wait_instructions   : list[str]   – SET_INTERLOCK / WAIT_INTERLOCK lines (may be empty)
            collision_detected  : bool
            proximity_warning   : bool
        """
        station_free_time = self.tracker.get_free_at(station_no)
        blocking_wagon    = self.tracker.get_occupied_by(station_no)

        # ── KEY RULE ────────────────────────────────────────────────────────
        # A WAIT is only needed when ANOTHER wagon is currently occupying the
        # station.  A wagon never waits for a station it already reserved
        # itself (that would be a false collision from the sequential pair
        # processing in gap_analysis_sequence).
        # ─────────────────────────────────────────────────────────────────────
        # Rule: A wagon must wait for the station to be free, regardless of who reserved it.
        # (This ensures the dip_time/task completes before any subsequent lift/access)
        if station_free_time > wagon_arrival_time:
            # ⭐ Use the canonical compute_entry_time() formula
            actual_entry_time, wait_time = compute_entry_time(
                wagon_arrival_time, station_free_time
            )
        else:
            # Station already free — arrive naturally, no wait
            actual_entry_time = wagon_arrival_time
            wait_time         = 0.0

        collision_detected = wait_time > 0   # True only for cross-wagon conflicts

        # ── SET_INTERLOCK vs WAIT_INTERLOCK pattern ────────────────────────────────────
        # wait_time > WAIT_THRESHOLD (5 s)  → SET_INTERLOCK  (exact known duration)
        # 0 < wait_time ≤ WAIT_THRESHOLD    → WAIT_INTERLOCK  (poll PLC free signal)
        # ─────────────────────────────────────────────────────────────────────
        # 🔥 Pattern: Similar implement everywhere (Rule 2: WAIT_INTERLOCK marker + SET_INTERLOCK duration)
        # ONLY for cross-wagon conflicts (shared stations)
        wait_instructions = []
        is_cross_wagon = (blocking_wagon is not None) and (blocking_wagon != wagon_id)
        
        if wait_time > 0 and is_cross_wagon:
            # Step 1: Insert WAIT_INTERLOCK as a reason marker
            reason = f"STATION_{station_no}_OCCUPIED_BY_{blocking_wagon}"
            wait_instructions.append(f"WAIT_INTERLOCK({reason})")
            
            # Step 2: Insert SET_INTERLOCK for the exact duration (if significant)
            if wait_time > self.WAIT_THRESHOLD_SEC:
                wait_instructions.append(f"SET_INTERLOCK({int(round(wait_time))})")
            else:
                # Fallback for short waits: just use the free signal
                wait_instructions.append(f"WAIT_INTERLOCK(STATION_{station_no}_FREE)")

        # Proximity / movement buffer check (secondary safety, different wagons only)
        # If two wagons are physically closer than safe_distance_mm AND their
        # next station is the same → insert WAIT_INTERLOCK(PROXIMITY_BUFFER).
        proximity_warning = False
        prox_instr = []
        if distance_mm is not None:
            for other_wagon, other_pos in self.wagon_positions.items():
                if other_wagon != wagon_id:
                    gap = abs(distance_mm - other_pos)
                    if gap < self.safe_distance_mm:
                        proximity_warning = True
                        prox_instr.append(
                            f"WAIT_INTERLOCK(PROXIMITY_BUFFER_{wagon_id}_vs_{other_wagon})"
                        )

        # Proximity instructions go first so the wagon clears clearance before
        # attempting to enter the station.
        all_instructions = prox_instr + wait_instructions

        # Commit the reservation so later wagons see the updated free_at
        self.tracker.reserve(station_no, wagon_id, actual_entry_time, process_time)

        # Update position tracker
        if distance_mm is not None:
            self.wagon_positions[wagon_id] = distance_mm

        return {
            "actual_entry_time": actual_entry_time,
            "wait_time":         wait_time,
            "wait_instructions": all_instructions,
            "collision_detected": collision_detected,
            "proximity_warning":  proximity_warning,
        }

    def release(self, station_no):
        """ Release the station (after GET FROM). """
        self.tracker.release(station_no)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def collision_report(self):
        """ Return a human-readable summary of the station occupancy table. """
        lines = ["\n[COLLISION ENGINE] Station Occupancy Table:"]
        lines.append(f"  {'Station':>8}  {'FreeAt(s)':>10}  {'OccupiedBy':>12}")
        for stn, info in sorted(self.tracker.station_status.items()):
            lines.append(
                f"  {stn:>8}  {info['free_at']:>10.1f}  {str(info['occupied_by']):>12}"
            )
        return "\n".join(lines)


# ==========================================
# POST-GENERATION COLLISION CORRECTION PASS
# ==========================================

def run_collision_correction_pass(sequence_data, station_list=None, config=None):
    """
    🚀 Best Implementation Strategy – Step 3:
       "Generate raw sequence → Run collision correction pass →
        Insert WAIT automatically → Recalculate timing."

    Walks the already-built sequence_data list (list of lists, first row = headers)
    and inserts missing SET_INTERLOCK / WAIT_INTERLOCK rows wherever two wagons would
    collide at the same station.

    This is a SECOND PASS safety net on top of the real-time engine inside
    gap_analysis_sequence.  It catches any timing edge-cases that slipped through.

    Algorithm
    ---------
    1. Build station_status from station_list (pre-init) or discover stations
       from the sequence itself.
    2. Walk rows in order (skipping existing WAIT rows).
    3. For every station-entry row:
           actual_entry_time = max(wagon_time_so_far, station_status[stn]["free_at"])
           wait_time         = actual_entry_time - wagon_time_so_far
    4. If wait_time > 5  → prepend WAIT_SEC(wait_time) row.
       If 0 < wait_time ≤ 5 → prepend WAIT_INT(STATION_n_FREE) row.
    5. Reserve station: free_at = actual_entry_time + travel_time.

    Parameters
    ----------
    sequence_data : list of lists
        First row must be headers; subsequent rows are data rows from
        gap_analysis_sequence.
    station_list  : iterable of int, optional
        Pre-known station numbers for tracker initialisation.
        If None, stations are discovered from the sequence.

    Returns
    -------
    list of lists – corrected sequence with WAIT rows injected where needed.
    """
    if not sequence_data or len(sequence_data) < 2:
        return sequence_data

    headers = sequence_data[0]
    rows    = sequence_data[1:]

    # Identify column indices from headers (tolerate column name variants)
    def _col(name, fallback=-1):
        try:
            return headers.index(name)
        except ValueError:
            return fallback

    col_wagon   = _col("Wagon")
    col_cmd     = _col("Command")
    col_val     = _col("Value")
    col_tt      = _col("TravelTime")
    col_acc     = _col("AccumulatedTime")
    col_row     = _col("Row", 0)
    col_step    = _col("Step No")
    col_cflag   = _col("CollisionFlag", -1)
    col_wt      = _col("WaitTime", -1)
    has_extra   = col_cflag != -1   # True when collision columns are present

    # Skip rows that are already WAIT instructions – don't re-evaluate them
    WAIT_COMMANDS = {"SET_INTERLOCK", "WAIT_INTERLOCK"}

    # Identify Shared Stations
    # Note: identify_shared_stations expects 'tanks' as list of dicts.
    # In correction pass, 'rows' are list of lists.
    
    # Fallback/Discovery logic integrated into run_collision_correction_pass
    station_to_wagons = defaultdict(set)
    for r in rows:
        try:
            stn = int(r[col_val]) if col_val != -1 else 0
            wid = r[col_wagon] if col_wagon != -1 else None
            if wid and stn: 
                station_to_wagons[stn].add(wid)
        except: continue
    shared_stations = {s for s, w in station_to_wagons.items() if len(w) >= 2}

    # ── 1. Pre-init station tracker ─────────────────────────────────────────
    tracker = StationOccupancyTracker(shared_stations=shared_stations)
    if station_list:
        tracker.initialize_from_stations(station_list)
    else:
        # Discover stations from sequence
        for r in rows:
            try:
                stn = int(r[col_val]) if col_val != -1 else 0
                tracker._ensure(stn)
            except (ValueError, TypeError, IndexError):
                pass

    # ── 2. Walk rows; per-wagon accumulated time ─────────────────────────────
    wagon_times = {}   # wagon_id → current accumulated time
    corrected   = [headers]
    WAIT_THRESHOLD = CollisionEngine.WAIT_THRESHOLD_SEC

    for r in rows:
        try:
            wagon_id = r[col_wagon] if col_wagon != -1 else None
            cmd      = r[col_cmd]   if col_cmd   != -1 else ""
            stn      = int(r[col_val]) if col_val != -1 else 0
            tt       = float(r[col_tt])  if col_tt  != -1 else 0.0
            acc_now  = float(r[col_acc]) if col_acc != -1 else 0.0
        except (ValueError, TypeError, IndexError):
            corrected.append(r)
            continue

        # Skip existing WAIT rows – they were already inserted by the engine
        if any(cmd.startswith(w) for w in WAIT_COMMANDS):
            corrected.append(r)
            if wagon_id:
                wagon_times[wagon_id] = acc_now + (float(r[col_tt]) if col_tt != -1 else 0.0)
            continue

        # Track per-wagon time
        if wagon_id not in wagon_times:
            wagon_times[wagon_id] = acc_now

        wagon_arrival = wagon_times.get(wagon_id, acc_now)
        station_free  = tracker.get_free_at(stn)
        blocking      = tracker.get_occupied_by(stn)
        cross_conflict = (blocking is not None) and (blocking != wagon_id)

        # ── 3. Apply core formula (Only for shared stations) ──
        if stn in shared_stations and station_free > wagon_arrival:
            actual_entry, wait_time = compute_entry_time(wagon_arrival, station_free)
        else:
            actual_entry = wagon_arrival
            wait_time    = 0.0

        # ── 4. Insert WAIT row if needed (ONLY for cross-wagon conflicts) ─────
        if wait_time > 0 and cross_conflict:
            # Step 1: Marker Row
            reason = f"STATION_{stn}_OCCUPIED_BY_{blocking}"
            wait_row1 = [None] * len(headers)
            wait_row1[col_row]   = r[col_row] if col_row != -1 else 1
            wait_row1[col_wagon] = wagon_id
            wait_row1[col_step]  = 0
            wait_row1[col_cmd]   = f"WAIT_INTERLOCK({reason})"
            wait_row1[col_val]   = stn
            if col_tt  != -1: wait_row1[col_tt]  = "0.00"
            if col_acc != -1: wait_row1[col_acc] = f"{wagon_arrival:.2f}"
            if has_extra:
                wait_row1[col_cflag] = "COLLISION"
                wait_row1[col_wt]    = f"{wait_time:.2f}"
            corrected.append(wait_row1)

            # Step 2: Duration Row (if significant)
            if wait_time > WAIT_THRESHOLD:
                wait_instr   = f"SET_INTERLOCK({int(round(wait_time))})"
                wait_sec_val = wait_time
            else:
                wait_instr   = f"WAIT_INTERLOCK(STATION_{stn}_FREE)"
                wait_sec_val = 0.0

            wait_row2 = list(wait_row1)
            wait_row2[col_cmd] = wait_instr
            if col_tt != -1: wait_row2[col_tt] = f"{wait_sec_val:.2f}"
            corrected.append(wait_row2)
            
            wagon_times[wagon_id] = wagon_arrival + wait_sec_val
        elif wait_time > 0:
            # Same wagon dip wait — no row, but advance clock
            wagon_times[wagon_id] = actual_entry

        # Reserve station for this wagon
        tracker.reserve(stn, wagon_id, actual_entry, tt)
        # Advance wagon clock by travel + process
        wagon_times[wagon_id] = actual_entry + tt

        corrected.append(r)

    return corrected


# ==========================================
# SEQUENCE GENERATION (PHYSICS-BASED)
# ==========================================
def calculate_time_value(distance1, distance2, distance3, sfspeed, fspeed, sspeed):
    try:
        sfs = sfspeed * 16.66 if sfspeed > 0 else 1.0
        fs = fspeed * 16.66 if fspeed > 0 else 1.0
        ss = sspeed * 16.66 if sspeed > 0 else 1.0
        return (distance1 / sfs) + (distance2 / fs) + (distance3 / ss)
    except: return 0

def build_wagon_routes(tanks, config):
    """
    Partition the flat tanks list into per-wagon ordered station lists.

    Each station belongs to exactly ONE wagon (determined by row + station range).
    Cross-row boundary entries that do not map to any wagon are silently skipped.

    Returns
    -------
    dict  {wagon_id: [tank_dict, tank_dict, ...]}
          Each list is in the same order as the input tanks list,
          containing only the stations owned by that wagon.
    """
    routes = {w: [] for w in config}

    for tank in tanks:
        try:
            s_no  = int(tank.get('station_no', 0))
            t_row = int(tank.get('Row') or tank.get('row') or 1)
        except (ValueError, TypeError):
            continue

        for w_name, w_cfg in config.items():
            w_row = int(w_cfg.get('row', 1))
            if w_row == t_row and w_cfg['min_stn'] <= s_no <= w_cfg['max_stn']:
                routes[w_name].append(tank)
                break   # each station → only one wagon

    # Drop wagons with no stations
    return {w: r for w, r in routes.items() if r}


def _insert_wait_rows(sequence_data, engine, wagon_id, station_no,
                      wagon_arrival_time, process_time, distance_mm,
                      t_row, step_counter, acc_time,
                      collision_counts):
    """
    Core collision formula (used identically for every station entry):

        actual_entry_time = max(wagon_arrival_time, station_free_time)
        wait_time         = actual_entry_time - wagon_arrival_time

    If wait_time > 0  AND the station is occupied by a DIFFERENT wagon:
        wait_time > 5 s  → SET_INTERLOCK(wait_time)
        else             → WAIT_INTERLOCK(STATION_<n>_FREE)

    Also checks proximity buffer against other wagons (secondary safety).

    Returns (actual_entry_time, step_counter, acc_time, collision_counts)
    """
    result = engine.compute_entry(
        wagon_id           = wagon_id,
        station_no         = station_no,
        wagon_arrival_time = wagon_arrival_time,
        process_time       = process_time,
        distance_mm        = distance_mm,
    )

    for wait_instr in result["wait_instructions"]:
        step_counter += 1
        wait_sec_val = 0
        if wait_instr.startswith("SET_INTERLOCK"):
            try:
                wait_sec_val = int(wait_instr.split("(")[1].rstrip(")"))
            except Exception:
                wait_sec_val = int(round(result["wait_time"]))

        flag = "COLLISION" if result["collision_detected"] else "PROXIMITY"
        sequence_data.append([
            t_row, wagon_id, step_counter,
            wait_instr, station_no,
            f"{wait_sec_val:.2f}",
            f"{acc_time:.2f}",
            flag,
            f"{result['wait_time']:.2f}",
        ])
        acc_time += wait_sec_val

    if result["collision_detected"]:
        collision_counts[0] += 1
    if result["proximity_warning"]:
        collision_counts[1] += 1

    return result["actual_entry_time"], step_counter, acc_time, collision_counts


def gap_analysis_sequence(tanks, config, enable_collision_engine=True,
                          safe_distance_mm=None):
    """
    Generates a unified sequence for all wagons in a single table.

    Anti-Collision Layer (NEW)
    --------------------------
    When enable_collision_engine=True (default), the function:
      1. Maintains a StationOccupancyTracker so each station records when it
         becomes free and which wagon is occupying it.
      2. Before every station entry, checks if a collision would occur.
      3. Inserts SET_INTERLOCK(<t>) or WAIT_INTERLOCK(STATION_<n>_FREE) rows automatically.
      4. Applies a proximity movement-buffer check (secondary safety).
      5. Recalculates accumulated timing so downstream steps stay correct.

    Args:
        tanks            : list of tank/station dicts
        config           : wagon config dict (from load_wagon_config)
        enable_collision_engine : bool – set False to use legacy behaviour
        safe_distance_mm : float – proximity buffer threshold (mm)

    Returns:
        List of lists [headers, row1, row2, ...]
        Collision-annotated rows carry Command values like SET_INTERLOCK / WAIT_INTERLOCK.
    """
    # Headers include extra collision columns so the table is self-documenting.
    if enable_collision_engine:
        headers = ["Row", "Wagon", "Step No", "Command", "Value",
                   "TravelTime", "AccumulatedTime", "CollisionFlag", "WaitTime"]
    else:
        headers = ["Row", "Wagon", "Step No", "Command", "Value",
                   "TravelTime", "AccumulatedTime"]

    sequence_data = [headers]
    accumulated_times = {w_id: 0.0 for w_id in config}
    step_counters    = {w_id: 0   for w_id in config}
    last_wagon_in_loop = None

    # Identify Shared Stations (STATIONS used by 2 or more wagons)
    shared_stations = identify_shared_stations(tanks, config) if enable_collision_engine else set()

    # Initialise the collision engine (shared across all wagons)
    if enable_collision_engine:
        engine = CollisionEngine(safe_distance_mm=safe_distance_mm, shared_stations=shared_stations)
        collision_count  = 0
        proximity_count  = 0
        
        # NEW: Full cycle collision tracking
        wagon_status = {w_id: {"completion_time": 0.0} for w_id in config}
        # Determine logical order: Row, then starting station
        wagon_order = sorted(config.keys(), key=lambda x: (config[x]['row'], config[x]['min_stn']))
        wagon_is_first_step = {w_id: True for w_id in config}
        # Track the very last station each wagon "PUT ON"
        wagon_last_destination = {w_id: None for w_id in config}

    # 🔥 RULE 1: CREATE GLOBAL STATION SIGNAL TABLE (Only For Shared Stations)
    all_stations = sorted(list(set(int(t.get('station_no', 0)) for t in tanks)))
    station_signal      = {s: False for s in all_stations if s in shared_stations}
    station_occupied_by = {s: None  for s in all_stations if s in shared_stations}
    station_free_time   = {s: 0.0   for s in all_stations if s in shared_stations}
    
    # Pre-build wagon routes to identify the final destination of each wagon
    wagon_routes = build_wagon_routes(tanks, config)

    print(f"\n[INFO] Generating unified row-aware sequence for {len(config)} wagons"
          f" [collision_engine={'ON' if enable_collision_engine else 'OFF'}]...")

    for i in range(len(tanks) - 1):
        curr_tank = tanks[i]
        next_tank = tanks[i + 1]

        try:
            s_curr = int(curr_tank.get('station_no', 0))
            s_next = int(next_tank.get('station_no', 0))
            t_row  = int(curr_tank.get('Row') or curr_tank.get('row') or 1)
        except ValueError:
            continue

        # Identify which wagon owns this station (by row + station range)
        wagon_id = None
        for w_name, w_cfg in config.items():
            w_row = int(w_cfg.get('row', 1))
            if w_row == t_row and w_cfg['min_stn'] <= s_curr <= w_cfg['max_stn']:
                wagon_id = w_name
                break

        if not wagon_id:
            continue

        # 🔥 NEW: Detect wagon transition to close the previous wagon's block
        # (Ensures Wagon 1's final GET FROM appears before Wagon 2's start)
        if last_wagon_in_loop is not None and wagon_id != last_wagon_in_loop:
            prev_w = last_wagon_in_loop
            prev_route = wagon_routes.get(prev_w, [])
            if prev_route:
                # Use tracking to find the last station actually reached
                p_last_stn = wagon_last_destination.get(prev_w)
                if p_last_stn is None: 
                    p_last_stn = int(prev_route[-1].get('station_no', 0))
                
                # Fetch row from route metadata
                p_row      = int(prev_route[-1].get('Row') or prev_route[-1].get('row') or 1)
                pw         = config.get(prev_w, {})
                plift      = float(pw.get('lift_time', 0))
                pacc       = accumulated_times.get(prev_w, 0.0)

                step_counters[prev_w] += 1
                if enable_collision_engine:
                    sequence_data.append([
                        p_row, prev_w, step_counters[prev_w], "GET FROM", p_last_stn,
                        f"{plift:.2f}", f"{pacc:.2f}", "-", "0.00"
                    ])
                else:
                    sequence_data.append([
                        p_row, prev_w, step_counters[prev_w], "GET FROM", p_last_stn,
                        f"{plift:.2f}", f"{pacc:.2f}"
                    ])
                accumulated_times[prev_w] = pacc + plift
                
                # 🔥 RULE 6: RELEASE STATION (Transition)
                if p_last_stn in shared_stations:
                    station_signal[p_last_stn] = False
                    station_occupied_by[p_last_stn] = None
        
        last_wagon_in_loop = wagon_id

        # 🔥 ADDED FULL CYCLE COLLISION PREVENTION (Rule 1, 2, 5)
        # Check if this is the very first time we encounter this wagon in the sequence.
        # If so, it must wait for the previous wagon in the logical order to FINISH.
        if enable_collision_engine and wagon_is_first_step.get(wagon_id):
            wagon_idx = wagon_order.index(wagon_id)
            if wagon_idx > 0:
                prev_wagon = wagon_order[wagon_idx - 1]
                # Rule 1: Wagon N cannot start until Wagon N-1 completion time
                prev_completion = wagon_status[prev_wagon]["completion_time"]
                current_arrival = accumulated_times[wagon_id]
                
                if current_arrival < prev_completion:
                    wait_time = prev_completion - current_arrival
                    
                    # Rule: If the startup collision is at a shared station, use station reason
                    prev_last_stn = wagon_last_destination.get(prev_wagon)
                    if prev_last_stn is not None and int(prev_last_stn) == s_curr:
                        reason = f"STATION_{s_curr}_OCCUPIED_BY_{prev_wagon}"
                    else:
                        reason = f"PROXIMITY_BUFFER_{wagon_id}_vs_{prev_wagon}"

                    # Rule 2: Insert automatic WAITs
                    step_counters[wagon_id] += 1
                    sequence_data.append([
                        t_row, wagon_id, step_counters[wagon_id], 
                        f"WAIT_INTERLOCK({reason})",
                        s_curr, "0.00", f"{current_arrival:.2f}",
                        "COLLISION", f"{wait_time:.2f}"
                    ])
                    
                    step_counters[wagon_id] += 1
                    sequence_data.append([
                        t_row, wagon_id, step_counters[wagon_id],
                        f"SET_INTERLOCK({int(round(wait_time))})",
                        s_curr, f"{wait_time:.2f}", f"{current_arrival:.2f}",
                        "COLLISION", f"{wait_time:.2f}"
                    ])
                    
                    # Rule 3: Update current time for this wagon
                    accumulated_times[wagon_id] += wait_time
                    
            wagon_is_first_step[wagon_id] = False

        w         = config[wagon_id]
        dist_curr = float(curr_tank.get('distance_mm', 0))
        dist_next = float(next_tank.get('distance_mm', 0))
        dist_total = abs(dist_next - dist_curr)
        process_time = float(curr_tank.get('dip_time_sec', 0))

        # Physics parameters
        distance3  = 500.0          # slow-speed approach zone (mm)
        distance2  = max(0, dist_total - distance3)
        distance1  = dist_total

        travel_time = calculate_time_value(
            distance1, distance2, distance3, w['sf'], w['f'], w['s']
        )
        acc_time = accumulated_times[wagon_id]

        # ----------------------------------------------------------------
        # ANTI-COLLISION CHECK for the current station (s_curr)
        # ----------------------------------------------------------------
        if enable_collision_engine:
            wagon_arrival = acc_time  # wagon arrives at s_curr at acc_time
            result = engine.compute_entry(
                wagon_id        = wagon_id,
                station_no      = s_curr,
                wagon_arrival_time = wagon_arrival,
                process_time    = process_time,
                distance_mm     = dist_curr,
            )

            # Insert WAIT rows if needed
            for wait_instr in result["wait_instructions"]:
                step_counters[wagon_id] += 1
                # Determine wait duration for SET_INTERLOCK rows (0 for WAIT_INTERLOCK)
                wait_sec_val = 0
                if wait_instr.startswith("SET_INTERLOCK"):
                    # Extract value from SET_INTERLOCK(n)
                    try:
                        wait_sec_val = int(wait_instr.split("(")[1].rstrip(")"))
                    except Exception:
                        wait_sec_val = int(round(result["wait_time"]))

                sequence_data.append([
                    t_row, wagon_id, step_counters[wagon_id],
                    wait_instr, s_curr,
                    f"{wait_sec_val:.2f}",
                    f"{acc_time:.2f}",
                    "COLLISION" if result["collision_detected"] else "PROXIMITY",
                    f"{result['wait_time']:.2f}"
                ])
                # Advance accumulated time by the wait
                acc_time += wait_sec_val

            # Update acc_time to the actual entry time
            # (max of original arrival and station free_at)
            acc_time = result["actual_entry_time"]

            if result["collision_detected"]:
                collision_count += 1
            if result["proximity_warning"]:
                proximity_count += 1

        # ----------------------------------------------------------------
        # 🔥 RULE 3, 4, 5: BEFORE ANY WAGON DOES "GET FROM" (Only For Shared Stations)
        # ----------------------------------------------------------------
        if s_curr in shared_stations and station_signal.get(s_curr) == True and station_occupied_by.get(s_curr) != wagon_id:
            other_wagon = station_occupied_by.get(s_curr)
            wait_time = max(0, station_free_time.get(s_curr, 0) - acc_time)
            
            # Rule 4: Insert WAIT_INTERLOCK SIGNAL
            step_counters[wagon_id] += 1
            sequence_data.append([
                t_row, wagon_id, step_counters[wagon_id],
                f"WAIT_INTERLOCK(STATION_{s_curr}_OCCUPIED_BY_{other_wagon})", s_curr,
                "0.00", f"{acc_time:.2f}", "COLLISION", f"{wait_time:.2f}"
            ])
            
            # Rule 5: Calculate SET_INTERLOCK TIME
            if wait_time > 0:
                step_counters[wagon_id] += 1
                sequence_data.append([
                    t_row, wagon_id, step_counters[wagon_id],
                    f"SET_INTERLOCK({int(round(wait_time))})", s_curr,
                    f"{wait_time:.2f}", f"{acc_time:.2f}", "COLLISION", f"{wait_time:.2f}"
                ])
                acc_time += wait_time
                accumulated_times[wagon_id] = acc_time

        # ----------------------------------------------------------------
        # GET FROM  (lift from current station)
        # ----------------------------------------------------------------
        step_counters[wagon_id] += 1
        if enable_collision_engine:
            sequence_data.append([
                t_row, wagon_id, step_counters[wagon_id], "GET FROM", s_curr,
                f"{w['lift_time']:.2f}", f"{acc_time:.2f}",
                "-", "0.00"
            ])
        else:
            sequence_data.append([
                t_row, wagon_id, step_counters[wagon_id], "GET FROM", s_curr,
                f"{w['lift_time']:.2f}", f"{acc_time:.2f}"
            ])
        acc_time += w['lift_time']
        
        # 🔥 RULE 6: WHEN WAGON LEAVES STATION (Release After GET FROM)
        # (Only For Shared Stations)
        if s_curr in shared_stations:
            station_signal[s_curr] = False
            station_occupied_by[s_curr] = None
            if enable_collision_engine:
                engine.release(s_curr)

        # ----------------------------------------------------------------
        # ANTI-COLLISION CHECK & RESERVATION for the destination station (s_next)
        # (Must happen BEFORE PUT ON for shared stations)
        # ----------------------------------------------------------------
        if enable_collision_engine:
            dest_arrival = acc_time + travel_time  # will arrive at s_next after travel
            dest_process = float(next_tank.get('dip_time_sec', 0))
            dest_result = engine.compute_entry(
                wagon_id           = wagon_id,
                station_no         = s_next,
                wagon_arrival_time = dest_arrival,
                process_time       = dest_process,
                distance_mm        = dist_next,
            )

            for wait_instr in dest_result["wait_instructions"]:
                step_counters[wagon_id] += 1
                wait_sec_val = 0
                if wait_instr.startswith("SET_INTERLOCK"):
                    try:
                        wait_sec_val = int(wait_instr.split("(")[1].rstrip(")"))
                    except Exception:
                        wait_sec_val = int(round(dest_result["wait_time"]))

                sequence_data.append([
                    t_row, wagon_id, step_counters[wagon_id],
                    wait_instr, s_next,
                    f"{wait_sec_val:.2f}",
                    f"{acc_time:.2f}",
                    "COLLISION" if dest_result["collision_detected"] else "PROXIMITY",
                    f"{dest_result['wait_time']:.2f}"
                ])
                acc_time += wait_sec_val

            if dest_result["collision_detected"]:
                collision_count += 1
            if dest_result["proximity_warning"]:
                proximity_count += 1
                
        # 🔥 RULE 2: RESERVE STATION BEFORE PUT ON
        # (Only For Shared Stations)
        if s_next in shared_stations:
            station_signal[s_next] = True
            station_occupied_by[s_next] = wagon_id
            # Use next_tank's dip time (the time it will spend in s_next)
            dest_process_time = float(next_tank.get('dip_time_sec', 0))
            # The station will be free after PUT ON completes + dip time
            station_free_time[s_next] = acc_time + travel_time + w['lower_time'] + dest_process_time

        # ----------------------------------------------------------------
        # PUT ON  (lower onto destination station)
        # ----------------------------------------------------------------
        step_counters[wagon_id] += 1
        if enable_collision_engine:
            sequence_data.append([
                t_row, wagon_id, step_counters[wagon_id], "PUT ON", s_next,
                f"{(travel_time + w['lower_time']):.2f}", f"{acc_time:.2f}",
                "-", "0.00"
            ])
        else:
            sequence_data.append([
                t_row, wagon_id, step_counters[wagon_id], "PUT ON", s_next,
                f"{(travel_time + w['lower_time']):.2f}", f"{acc_time:.2f}"
            ])
        acc_time += travel_time + w['lower_time']

        accumulated_times[wagon_id] = acc_time
        wagon_last_destination[wagon_id] = s_next

        # 🔥 Rule 4: Proactive Completion Time Calculation
        # Identify if this was the final destination (last hop) for this wagon.
        # If so, calculate completion time (including return-to-base) so the NEXT
        # wagon in the sequence can use it for its startup WAIT.
        if wagon_id in wagon_routes and next_tank == wagon_routes[wagon_id][-1]:
            last_tank = next_tank
            last_stn_pos = float(last_tank.get('distance_mm', 0))
            basic_stn = w.get('basic_pos', 0)
            
            # Find base position distance
            basic_dist = 0.0
            for t in tanks:
                if int(t.get('station_no', 0)) == basic_stn:
                    basic_dist = float(t.get('distance_mm', 0))
                    break
            
            dist_to_base = abs(basic_dist - last_stn_pos)
            d3_b = 500.0
            d2_b = max(0, dist_to_base - d3_b)
            d1_b = dist_to_base
            travel_to_base = calculate_time_value(d1_b, d2_b, d3_b, w['sf'], w['f'], w['s'])
            
            # Completion includes: Last PUT ON + Final LIFT (from s_next) + TravelToBase + LowerTime
            # (multiplier applied at the very end)
            cycle_finish = acc_time + w.get('lift_time', 0) + travel_to_base + w.get('lower_time', 0)
            final_completion = cycle_finish * w.get('occupy_factor', 1.0)
            wagon_status[wagon_id]["completion_time"] = final_completion

    # ----------------------------------------------------------------
    # FINAL GET FROM — closing lift for the very last wagon in the loop
    # ----------------------------------------------------------------
    if last_wagon_in_loop:
        w_name = last_wagon_in_loop
        route  = wagon_routes.get(w_name, [])
        if route:
            last_stn = wagon_last_destination.get(w_name)
            if last_stn is None: last_stn = int(route[-1].get('station_no', 0))
            
            t_row     = int(route[-1].get('Row') or route[-1].get('row') or 1)
            w         = config.get(w_name, {})
            lift_time = float(w.get('lift_time', 0))
            acc_time  = accumulated_times.get(w_name, 0.0)

            step_counters[w_name] += 1
            if enable_collision_engine:
                sequence_data.append([
                    t_row, w_name, step_counters[w_name], "GET FROM", last_stn,
                    f"{lift_time:.2f}", f"{acc_time:.2f}", "-", "0.00"
                ])
            else:
                sequence_data.append([
                    t_row, w_name, step_counters[w_name], "GET FROM", last_stn,
                    f"{lift_time:.2f}", f"{acc_time:.2f}"
                ])
            accumulated_times[w_name] = acc_time + lift_time
            
            # 🔥 RULE 6: RELEASE STATION (Absolute Final)
            if last_stn in shared_stations:
                station_signal[last_stn] = False
                station_occupied_by[last_stn] = None

    # Print collision engine summary
        
    # Print collision engine summary
    if enable_collision_engine:
        print(engine.collision_report())
        print(f"\n[COLLISION ENGINE] Summary:")
        print(f"  Total station-collision WAITs inserted : {collision_count}")
        print(f"  Total proximity-buffer  WAITs inserted : {proximity_count}")
        print(f"  Total WAIT instructions total          : {collision_count + proximity_count}")

    # ── POST-GENERATION COLLISION CORRECTION PASS ──────────────────────────
    # A second pass over the already-built sequence to catch any timing edge-
    # cases that slipped through the per-step engine above.  Uses the same
    # compute_entry_time() formula and the same SET_INTERLOCK / WAIT_INTERLOCK rules.
    if enable_collision_engine:
        all_stations = list({
            int(t.get('station_no', 0))
            for t in tanks
            if t.get('station_no') is not None
        })
        sequence_data = run_collision_correction_pass(sequence_data, all_stations, config=config)

    return sequence_data

def generate_sequence_from_data(tanks_data, config=None, verbose=True):
    """
    Generate sequence from tanks data directly (list of dicts or pandas DataFrame).
    This function accepts data directly from GUI to avoid CSV file collisions.
    
    Args:
        tanks_data: List of dictionaries or pandas DataFrame with tank data
        config: Optional wagon config dict. If None, loads from default location.
        verbose: If True, prints progress messages
    
    Returns:
        List of lists representing the sequence [headers, row1, row2, ...]
    """
    try:
        # Convert DataFrame to list of dicts if needed
        try:
            # Try pandas DataFrame conversion
            if hasattr(tanks_data, 'to_dict') and hasattr(tanks_data, 'columns'):
                # It's a pandas DataFrame
                tanks = tanks_data.to_dict('records')
            elif isinstance(tanks_data, list):
                # Already a list of dicts
                tanks = tanks_data
            else:
                raise ValueError("tanks_data must be a list of dicts or pandas DataFrame")
        except AttributeError:
            # Fallback: if it's iterable, try to convert
            if isinstance(tanks_data, list):
                tanks = tanks_data
            else:
                raise ValueError("tanks_data must be a list of dicts or pandas DataFrame")
        
        if len(tanks) == 0:
            if verbose:
                print("Error: No tank data provided.")
            return None
        
        # Load wagon config if not provided
        if config is None:
            config = load_wagon_config()
            if not config:
                if verbose:
                    print("Error: No wagon configuration loaded.")
                return None
        
        # Generate Sequence
        if verbose:
            print(f"\nGenerating Unified Multi-Wagon Sequence from {len(tanks)} tanks...")
        
        unified_seq = gap_analysis_sequence(tanks, config)
        
        if verbose:
            if len(unified_seq) > 1:
                print(tabulate(unified_seq[1:], headers=unified_seq[0], tablefmt="grid"))
            else:
                print("\nNo moves could be assigned to any wagon based on their station ranges.")
        
        return unified_seq
        
    except Exception as e:
        if verbose:
            print(f"Error generating sequence: {e}")
            import traceback
            traceback.print_exc()
        raise

def generate_sequence_from_tanks(csv_path=None, tanks_data=None, speeds_input=None, config=None):
    """
    Generate sequence from CSV file or direct data input.
    Supports both CSV file path and direct data input for GUI integration.
    
    Args:
        csv_path: Path to CSV file (optional if tanks_data is provided)
        tanks_data: Direct data input - list of dicts or DataFrame (optional if csv_path is provided)
        speeds_input: Deprecated, kept for backward compatibility
        config: Optional wagon config dict. If None, loads from default location.
    
    Returns:
        List of lists representing the sequence [headers, row1, row2, ...]
    """
    # If tanks_data is provided, use direct data input
    if tanks_data is not None:
        return generate_sequence_from_data(tanks_data, config=config, verbose=True)
    
    # Otherwise, use CSV file (backward compatibility)
    if csv_path is None:
        csv_path = tanks_csv_default
    
    print(f"Processing sequence for {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return None

    # Load All Wagons
    if config is None:
        config = load_wagon_config()
    if not config:
        print("Error: No wagon configuration loaded.")
        return None
        
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            tanks = list(reader)
        
        # Generate Sequence
        return generate_sequence_from_data(tanks, config=config, verbose=True)

    except Exception as e:
        print(f"Error generating sequence: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_sequence_ai(csv_path, model_path, config=None):
    """
    Generates sequence using the trained RL model.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
    
    if csv_path is None:
        csv_path = tanks_csv_default
        
    print(f"Generating AI sequence using model: {model_path}")
    print(f"Input data: {csv_path}")
    
    # Load Environment Data
    train_data, adj_list, vocab_cmd = load_data()
    inv_vocab = {v: k for k, v in vocab_cmd.items()}
    
    # Load Wagon Config
    if config is None:
        config = load_wagon_config()
    if not config:
        print("Error: No wagon configuration.")
        return None

    # Load Model
    n_stations = MAX_STATIONS
    agent = PPOAgent(vocab_cmd, n_stations, adj_list)
    try:
        agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load Tanks
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            tanks = list(reader)
    except Exception as e:
        print(f"Error loading tanks: {e}")
        return None

    # Simulator / Inference Loop
    env = WagonEnv(adj_list, n_stations)
    state = env.reset()
    curr_cmd_seq = [vocab_cmd["<SOS>"]]
    curr_stn_seq = [0]
    
    # Output table structure
    # AI doesn't directly map to the physics-based log easily without simulation
    # but we can output what the AI picks.
    sequence_data = [["Row", "Wagon", "Step No", "Command", "Value", "TravelTime", "AccumulatedTime"]]
    
    # Simplified AI Step logic
    for t in range(len(tanks) * 2): # Heuristic: 2 steps per tank (GET/PUT)
        a_cmd, a_stn = agent.select_action(curr_cmd_seq, curr_stn_seq, state)
        next_state, reward, done = env.step(a_cmd, a_stn)
        
        cmd_name = inv_vocab.get(a_cmd, "UNKNOWN")
        
        # Identify wagon for display
        wagon_id = "AI_Wagon"
        for w_name, w_cfg in config.items():
            if w_cfg['min_stn'] <= a_stn <= w_cfg['max_stn']:
                wagon_id = w_name
                break
        
        # Dummy times for AI mode log (inference doesn't have physics integrated yet)
        sequence_data.append([
            1, wagon_id, t+1, cmd_name, a_stn, "0.00", "0.00"
        ])
        
        curr_cmd_seq.append(a_cmd)
        curr_stn_seq.append(a_stn)
        state = next_state
        if done or cmd_name == "<EOS>":
            break
            
    print(tabulate(sequence_data[1:], headers=sequence_data[0], tablefmt="grid"))
    return sequence_data

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    global wagon_config_csv, row_config_csv
    parser = argparse.ArgumentParser(description="Sequence Generation and AI Training Tool")
    parser.add_argument("--mode", type=str, choices=["gen", "train", "ai_gen"], default="gen", help="Execution mode")
    parser.add_argument("--input", type=str, help="Path to tanks CSV", default=tanks_csv_default)
    parser.add_argument("--speeds", type=str, help="Wagon Name (Optional, now uses config ranges)", default="Wagon 1")
    parser.add_argument("--config", type=str, help="Path to wagon config CSV", default=wagon_config_csv)
    parser.add_argument("--row_config", type=str, help="Path to row config CSV", default=row_config_csv)
    parser.add_argument("--model", type=str, help="Path to trained model (.pth)", default=os.path.join(parent_dir, "model_v9.pth"))
    
    args = parser.parse_args()
    wagon_config_csv = args.config
    row_config_csv = args.row_config

    if args.mode == "gen":
        generate_sequence_from_tanks(csv_path=args.input, speeds_input=args.speeds, config=load_wagon_config(wagon_config_csv))
    elif args.mode == "ai_gen":
        generate_sequence_ai(csv_path=args.input, model_path=args.model, config=load_wagon_config(wagon_config_csv))
    else:
        # AI Training Mode
        train_data, adj_list, vocab_cmd = load_data()
        if not train_data:
            print("No training data found. Exiting.")
            return

        agent = PPOAgent(vocab_cmd, MAX_STATIONS, adj_list)
        env = WagonEnv(adj_list, MAX_STATIONS)
        
        print("\nStarting RL Training Loop (1000 Episodes)...")
        for episode in range(1, 1001): 
            state = env.reset()
            curr_cmd_seq = [vocab_cmd["<SOS>"]]
            curr_stn_seq = [0]
            ep_reward = 0
            collisions = 0
            
            for t in range(MAX_STEPS):
                a_cmd, a_stn = agent.select_action(curr_cmd_seq, curr_stn_seq, state)
                next_state, reward, done = env.step(a_cmd, a_stn)
                
                # Buffer rewards and terminal flags
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)
                
                if reward < 0: collisions += 1
                
                curr_cmd_seq.append(a_cmd)
                curr_stn_seq.append(a_stn)
                state = next_state
                ep_reward += reward
                if done: break
            
            # Update the agent after each episode
            agent.update()
            
            if episode % 20 == 0:
                print(f"Episode {episode}: Total Reward = {ep_reward} (Collisions/Invalid: {collisions})")
            
        print("Training Finished. Model saved as 'model_v7.pth'")
        torch.save(agent.policy.state_dict(), "model_v7.pth")

if __name__ == "__main__":
    main()
