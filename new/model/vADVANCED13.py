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
        with open(config_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                w_num = (
                    row.get("Transporter Name") or row.get("Wagon Number") or ""
                ).strip()
                if not w_num:
                    continue

                config[w_num] = {
                    "sf": float(row.get("Superfast Speed", 0)),
                    "f": float(row.get("Fast Speed", 0)),
                    "s": float(row.get("Slow Speed", 0)),
                    "lift_time": float(row.get("Lift Time", 0)),
                    "lower_time": float(row.get("Lower Time", 0)),
                    "min_stn": int(row.get("Minimum Station No", 1)),
                    "max_stn": int(row.get("Maximum Station No", 200)),
                    "basic_pos": int(row.get("Basic Position", 0)),
                    "row": int(row.get("Row") or row.get("Row Number") or 1),
                    "occupy_factor": float(row.get("Station Occupy Factor", 1.0)),
                }
    except Exception as e:
        print(f"Error loading wagon config: {e}")
    return config


def load_row_config(config_path=row_config_csv):
    """Loads row configuration from CSV."""
    rows = {}
    if not os.path.exists(config_path):
        return rows
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_num = int(row.get("Row Number", 1))
                rows[r_num] = {
                    "min_stn": int(row.get("First Station No", 1)),
                    "max_stn": int(row.get("Last Station No", 200)),
                }
    except Exception as e:
        print(f"Error loading row config: {e}")
    return rows


def verify_stations(tanks, min_stn, max_stn):
    """
    Verifies if the total number of unique stations in the tank list matches
    the expected count (Maximum Station No).
    """
    present_stations = {
        tank.get("station_no") for tank in tanks if tank.get("station_no")
    }
    total_count = len(present_stations)
    expected_count = max_stn

    print(f"\n[STATION VERIFICATION]")
    print(f"Goal: Total stations should be {expected_count}")
    print(f"Status: Found {total_count} stations in the tank data.")

    if total_count == expected_count:
        print(
            f"SUCCESS: Total station count matches Maximum Station No ({expected_count})."
        )
        return True
    else:
        diff = expected_count - total_count
        status = "missing" if diff > 0 else "extra"
        print(
            f"FAILURE: Total station count ({total_count}) does not match Maximum Station No ({expected_count})."
        )
        print(f"There are {abs(diff)} {status} stations.")
        return False


def load_data():
    """Loads historical sequence data for training RL models."""
    print("Loading historical data for AI training...")
    sequences = defaultdict(list)
    vocab_cmd = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}

    if not os.path.exists(csv_programs):
        print(f"Warning: {csv_programs} not found. AI training will use empty data.")
        return [], {}, vocab_cmd

    with open(csv_programs, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # seq_id,project_id,wagon_no,step_no,command,station_no
                seq_id = row.get("seq_id") + "_" + row.get("project_id")
                cmd = row.get("command")
                stn = int(row.get("station_no")) if row.get("station_no") else 0
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
        with open(csv_zones, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s1 = int(row["Row1StationNo"])
                    s2 = int(row["Row2StationNo"])
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=N_HEADS, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.fc_cmd = nn.Linear(d_model, n_cmds)
        self.fc_stn = nn.Linear(d_model, n_stations)

    def forward(self, cmd_seq, stn_seq):
        seq_len = cmd_seq.size(1)
        pos = (
            torch.arange(seq_len, device=cmd_seq.device)
            .unsqueeze(0)
            .expand(cmd_seq.size(0), -1)
        )
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
        pos = (
            torch.arange(seq_len, device=cmd_seq.device)
            .unsqueeze(0)
            .expand(cmd_seq.size(0), -1)
        )
        x_seq = (
            self.transformer.cmd_emb(cmd_seq)
            + self.transformer.stn_emb(stn_seq)
            + self.transformer.pos_emb(pos)
        )
        x_seq = self.transformer.transformer(x_seq)
        ctx_seq = x_seq[:, -1, :]

        # Extract context from Safety GNN
        x_gnn = self.gnn(station_states, adj_matrix)
        ctx_safe, _ = torch.max(x_gnn, dim=1)

        state = torch.cat([ctx_seq, ctx_safe], dim=-1)
        return (
            torch.softmax(self.actor_cmd(state), dim=-1),
            torch.softmax(self.actor_stn(state), dim=-1),
            self.critic(state),
        )


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
            probs_cmd, probs_stn, val = self.policy_old(
                cmd_tensor, stn_tensor, state_tensor, self.adj_matrix
            )

        dist_cmd = Categorical(probs_cmd)
        dist_stn = Categorical(probs_stn)

        action_cmd = dist_cmd.sample()
        action_stn = dist_stn.sample()

        # Store in buffer (use the actual index of the current step for context)
        # We need to take the probability corresponding to the last ACTUAL step
        last_idx = len(cmd_seq) - 1

        self.buffer.cmd_seqs.append(torch.tensor(padded_cmd, dtype=torch.long))
        self.buffer.stn_seqs.append(torch.tensor(padded_stn, dtype=torch.long))
        self.buffer.station_states.append(
            torch.tensor(station_states, dtype=torch.long)
        )
        self.buffer.actions_cmd.append(action_cmd)
        self.buffer.actions_stn.append(action_stn)
        self.buffer.logprobs.append(
            dist_cmd.log_prob(action_cmd) + dist_stn.log_prob(action_stn)
        )

        return action_cmd.item(), action_stn.item()

    def update(self):
        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
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
            probs_cmd, probs_stn, state_values = self.policy(
                old_cmd_seqs, old_stn_seqs, old_station_states, self.adj_matrix
            )

            dist_cmd = Categorical(probs_cmd)
            dist_stn = Categorical(probs_stn)

            logprobs = dist_cmd.log_prob(old_actions_cmd) + dist_stn.log_prob(
                old_actions_stn
            )
            dist_entropy = dist_cmd.entropy() + dist_stn.entropy()
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

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
        if stn >= self.max_stations:
            return self.station_occupancy, -50, True
        if self.station_occupancy[stn] == 1:
            reward -= 50
        elif any(self.station_occupancy[n] == 1 for n in self.adj_list.get(stn, [])):
            reward -= 20
        self.station_occupancy[stn] = 1
        reward += 1
        if self.step_count >= MAX_STEPS:
            done = True
        return self.station_occupancy, reward, done


def filter_tanks_by_load_rotation(tanks, load_index=0):
    """
    Identifies logical process steps by grouping consecutive tanks with the same 'process_name'.
    For each step, it selects one physical station based on the load_index to enable
    load balancing across parallel tanks.

    This handles the scenario where 4-5 stations of the same solution are placed 'one by one'.
    """
    if not tanks:
        return []

    # 1. Group consecutive stations with the same process name into 'logical steps'
    logical_steps = []
    if tanks:
        current_step_options = [tanks[0]]
        for i in range(1, len(tanks)):
            prev_proc = (
                (current_step_options[-1].get("process_name") or "").strip().lower()
            )
            curr_proc = (tanks[i].get("process_name") or "").strip().lower()

            if curr_proc == prev_proc and curr_proc != "":
                current_step_options.append(tanks[i])
            else:
                logical_steps.append(current_step_options)
                current_step_options = [tanks[i]]
        logical_steps.append(current_step_options)

    # 2. Select one station per logical step based on load_index
    load_specific_tanks = []
    for step_options in logical_steps:
        # Load balancing: pick tank based on load_index modulo number of available tanks
        selected_idx = load_index % len(step_options)
        selected_tank = step_options[selected_idx]
        load_specific_tanks.append(selected_tank)

    return load_specific_tanks


def get_tank_criticality(tank):
    """
    Get criticality level from tank data.

    Args:
        tank: dict containing tank data

    Returns:
        str: "HIGH" or "LOW", defaults to "HIGH"
    """
    criticality = (
        (tank.get("criticality") or tank.get("critical_status") or "").strip().upper()
    )
    if criticality in ("HIGH", "LOW"):
        return criticality
    return "HIGH"


MAX_HOLD_TIME_LOW = 300


def filter_tanks_sequential(tanks, load_index=0):
    """
    Sequential processing: Tank 1 → Tank 2 → Tank 3
    Each load processes ALL tanks in order (not modulo-based selection).
    Multiple loads run INDEPENDENTLY in parallel.

    Args:
        tanks: list of tank dictionaries
        load_index: index of the load (used for offset timing, not tank selection)

    Returns:
        list: tanks in sequential order
    """
    if not tanks:
        return []

    logical_steps = []
    if tanks:
        current_step_options = [tanks[0]]
        for i in range(1, len(tanks)):
            prev_proc = (
                (current_step_options[-1].get("process_name") or "").strip().lower()
            )
            curr_proc = (tanks[i].get("process_name") or "").strip().lower()

            if curr_proc == prev_proc and curr_proc != "":
                current_step_options.append(tanks[i])
            else:
                logical_steps.append(current_step_options)
                current_step_options = [tanks[i]]
        logical_steps.append(current_step_options)

    sequential_tanks = []
    for step_options in logical_steps:
        for tank in step_options:
            sequential_tanks.append(tank)

    return sequential_tanks


def filter_tanks_grouped(tanks, load_index=0):
    """
    Group-based routing: Multiple source tanks feed into collector tanks.

    Logic:
    1. Group tanks by process_name
    2. For groups with multiple tanks: last tank is collector, all others are sources
    3. The collector feeds into the first tank of next group

    Example flow (expected):
    - 1 → 2, 1 → 3, 1 → 4 (Loading → A acid)
    - 2 → 5, 3 → 5, 4 → 5 (A acid → Post Rinse)
    - 5 → 6 (Post Rinse → B acid)
    - 6 → 9, 7 → 9, 8 → 9 (B acid → Rinse)
    - 9 → 10 (Rinse → Unloading)

    Returns:
        list: routing pairs [(from_stn, to_stn), ...]
    """
    if not tanks:
        return []

    tanks_sorted = sorted(tanks, key=lambda x: int(x.get("station_no", 0)))

    process_groups = []
    current_group = []
    current_process = None

    for tank in tanks_sorted:
        proc = (tank.get("process_name") or "").strip().lower()
        if current_process is None:
            current_process = proc
            current_group = [tank]
        elif proc == current_process and proc != "":
            current_group.append(tank)
        else:
            if current_group:
                process_groups.append(current_group)
            current_process = proc
            current_group = [tank]

    if current_group:
        process_groups.append(current_group)

    routes = []

    for i, group in enumerate(process_groups):
        if len(group) > 1:
            collector = group[-1]
            collector_stn = int(collector.get("station_no", 0))
            for source in group[:-1]:
                source_stn = int(source.get("station_no", 0))
                routes.append((source_stn, collector_stn))

            if i < len(process_groups) - 1:
                next_group = process_groups[i + 1]
                next_first_stn = int(next_group[0].get("station_no", 0))
                routes.append((collector_stn, next_first_stn))
        else:
            single_tank = group[0]
            single_stn = int(single_tank.get("station_no", 0))
            if i < len(process_groups) - 1:
                next_group = process_groups[i + 1]
                next_first_stn = int(next_group[0].get("station_no", 0))
                routes.append((single_stn, next_first_stn))

    return routes


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
                    stn = int(tank.get("station_no", 0))
                elif isinstance(tank, (list, tuple)):
                    # This branch is for internal calls where route might be rows
                    continue
                else:
                    continue
                station_to_wagons[stn].add(wagon_id)
            except (ValueError, TypeError, AttributeError):
                continue

    shared_stations = {
        stn for stn, wagons in station_to_wagons.items() if len(wagons) >= 2
    }
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
        self.shared_stations = (
            set(shared_stations) if shared_stations is not None else set()
        )

    # ------------------------------------------------------------------
    #  Initialisation helpers
    # ------------------------------------------------------------------
    def _ensure(self, station_no):
        """Lazily initialise a station entry on first access."""
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
        """Return the time (seconds) at which station_no becomes free."""
        self._ensure(station_no)
        return self.station_status[station_no]["free_at"]

    def get_occupied_by(self, station_no):
        """Return the wagon_id currently holding station_no (or None)."""
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
        self.station_status[station_no]["free_at"] = entry_time + process_time
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
        """Return the full station status table for logging/debugging."""
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
    wait_time = actual_entry_time - wagon_arrival_time
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

    WAIT_THRESHOLD_SEC = 5  # seconds – boundary between WAIT_SEC and WAIT_INT
    DEFAULT_SAFE_DISTANCE_MM = 600  # mm – minimum gap between wagons

    def __init__(self, safe_distance_mm=None, shared_stations=None):
        self.tracker = StationOccupancyTracker(shared_stations=shared_stations)
        self.safe_distance_mm = safe_distance_mm or self.DEFAULT_SAFE_DISTANCE_MM
        # {wagon_id: current_position_mm}
        self.wagon_positions = {}

    # ------------------------------------------------------------------
    # Core collision check
    # ------------------------------------------------------------------
    def compute_entry(
        self, wagon_id, station_no, wagon_arrival_time, process_time, distance_mm=None
    ):
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
        blocking_wagon = self.tracker.get_occupied_by(station_no)

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
            wait_time = 0.0

        collision_detected = wait_time > 0  # True only for cross-wagon conflicts

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
            "wait_time": wait_time,
            "wait_instructions": all_instructions,
            "collision_detected": collision_detected,
            "proximity_warning": proximity_warning,
        }

    def release(self, station_no):
        """Release the station (after GET FROM)."""
        self.tracker.release(station_no)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def collision_report(self):
        """Return a human-readable summary of the station occupancy table."""
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
    rows = sequence_data[1:]

    # Identify column indices from headers (tolerate column name variants)
    def _col(name, fallback=-1):
        try:
            return headers.index(name)
        except ValueError:
            return fallback

    col_wagon = _col("Wagon")
    col_cmd = _col("Command")
    col_val = _col("Value")
    col_tt = _col("TravelTime")
    col_acc = _col("AccumulatedTime")
    col_row = _col("Row", 0)
    col_step = _col("Step No")
    col_cflag = _col("CollisionFlag", -1)
    col_wt = _col("WaitTime", -1)
    has_extra = col_cflag != -1  # True when collision columns are present

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
        except:
            continue
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
    wagon_times = {}  # wagon_id → current accumulated time
    corrected = [headers]
    WAIT_THRESHOLD = CollisionEngine.WAIT_THRESHOLD_SEC

    for r in rows:
        try:
            wagon_id = r[col_wagon] if col_wagon != -1 else None
            cmd = r[col_cmd] if col_cmd != -1 else ""
            stn = int(r[col_val]) if col_val != -1 else 0
            tt = float(r[col_tt]) if col_tt != -1 else 0.0
            acc_now = float(r[col_acc]) if col_acc != -1 else 0.0
        except (ValueError, TypeError, IndexError):
            corrected.append(r)
            continue

        # Skip existing WAIT rows – they were already inserted by the engine
        if any(cmd.startswith(w) for w in WAIT_COMMANDS):
            corrected.append(r)
            if wagon_id:
                wagon_times[wagon_id] = acc_now + (
                    float(r[col_tt]) if col_tt != -1 else 0.0
                )
            continue

        # Track per-wagon time
        if wagon_id not in wagon_times:
            wagon_times[wagon_id] = acc_now

        wagon_arrival = wagon_times.get(wagon_id, acc_now)
        station_free = tracker.get_free_at(stn)
        blocking = tracker.get_occupied_by(stn)
        cross_conflict = (blocking is not None) and (blocking != wagon_id)

        # ── 3. Apply core formula (Only for shared stations) ──
        if stn in shared_stations and station_free > wagon_arrival:
            actual_entry, wait_time = compute_entry_time(wagon_arrival, station_free)
        else:
            actual_entry = wagon_arrival
            wait_time = 0.0

        # ── 4. Insert WAIT row if needed (ONLY for cross-wagon conflicts) ─────
        if wait_time > 0 and cross_conflict:
            # Step 1: Marker Row
            reason = f"STATION_{stn}_OCCUPIED_BY_{blocking}"
            wait_row1 = [None] * len(headers)
            wait_row1[col_row] = r[col_row] if col_row != -1 else 1
            wait_row1[col_wagon] = wagon_id
            wait_row1[col_step] = 0
            wait_row1[col_cmd] = f"WAIT_INTERLOCK({reason})"
            wait_row1[col_val] = stn
            if col_tt != -1:
                wait_row1[col_tt] = "0.00"
            if col_acc != -1:
                wait_row1[col_acc] = f"{wagon_arrival:.2f}"
            if has_extra:
                wait_row1[col_cflag] = "COLLISION"
                wait_row1[col_wt] = f"{wait_time:.2f}"
            corrected.append(wait_row1)

            # Step 2: Duration Row (if significant)
            if wait_time > WAIT_THRESHOLD:
                wait_instr = f"SET_INTERLOCK({int(round(wait_time))})"
                wait_sec_val = wait_time
            else:
                wait_instr = f"WAIT_INTERLOCK(STATION_{stn}_FREE)"
                wait_sec_val = 0.0

            wait_row2 = list(wait_row1)
            wait_row2[col_cmd] = wait_instr
            if col_tt != -1:
                wait_row2[col_tt] = f"{wait_sec_val:.2f}"
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
    except:
        return 0


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
            s_no = int(tank.get("station_no", 0))
            t_row = int(tank.get("Row") or tank.get("row") or 1)
        except (ValueError, TypeError):
            continue

        for w_name, w_cfg in config.items():
            w_row = int(w_cfg.get("row", 1))
            if w_row == t_row and w_cfg["min_stn"] <= s_no <= w_cfg["max_stn"]:
                routes[w_name].append(tank)
                break  # each station → only one wagon

    # Drop wagons with no stations
    return {w: r for w, r in routes.items() if r}


def _insert_wait_rows(
    sequence_data,
    engine,
    wagon_id,
    station_no,
    wagon_arrival_time,
    process_time,
    distance_mm,
    t_row,
    step_counter,
    acc_time,
    collision_counts,
):
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
        wagon_id=wagon_id,
        station_no=station_no,
        wagon_arrival_time=wagon_arrival_time,
        process_time=process_time,
        distance_mm=distance_mm,
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
        sequence_data.append(
            [
                t_row,
                wagon_id,
                step_counter,
                wait_instr,
                station_no,
                f"{wait_sec_val:.2f}",
                f"{acc_time:.2f}",
                flag,
                f"{result['wait_time']:.2f}",
            ]
        )
        acc_time += wait_sec_val

    if result["collision_detected"]:
        collision_counts[0] += 1
    if result["proximity_warning"]:
        collision_counts[1] += 1

    return result["actual_entry_time"], step_counter, acc_time, collision_counts


def gap_analysis_sequence(
    tanks,
    config,
    enable_collision_engine=True,
    safe_distance_mm=500.0,
    num_loads=1,
    processing_mode="sequential",
):
    """
    Advanced Multi-Load Interleaved Scheduler with Collision Avoidance.

    Args:
        tanks: list of tank dictionaries
        config: wagon configuration
        enable_collision_engine: whether to enable collision detection
        safe_distance_mm: safe distance between wagons
        num_loads: number of parallel loads
        processing_mode: "sequential" for sequential processing with criticality,
                        "load_balance" for original load rotation logic

    Logic:
    1. Pre-calculate paths for each load.
    2. Event-driven greedy dispatcher for GET->PUT moves.
    3. Collision Engine: Before any move, check if any other wagon is too close.
       If blocked, the wagon waits at its current position until the path is clear.

    Sequential Mode Additional Logic:
    - Each load processes tanks sequentially (Tank1→Tank2→Tank3)
    - HIGH criticality: items moved immediately after processing completes
    - LOW criticality: items can wait up to MAX_HOLD_TIME_LOW (5 min) before forced pickup
    """
    if safe_distance_mm is None:
        safe_distance_mm = 500.0

    headers = [
        "Row",
        "Wagon",
        "Step No",
        "Command",
        "Value",
        "TravelTime",
        "AccumulatedTime",
        "CollisionFlag",
        "WaitTime",
    ]
    sequence_data = [headers]

    # --- 1. Prepare Load Paths ---
    load_paths = []
    for l_idx in range(num_loads):
        if processing_mode == "sequential":
            path = filter_tanks_sequential(tanks, load_index=l_idx)
        else:
            path = filter_tanks_by_load_rotation(tanks, load_index=l_idx)
        load_paths.append(path)

    # --- 2. Initialize State ---
    load_states = []
    for l_idx in range(num_loads):
        if not load_paths[l_idx]:
            continue
        first_tank = load_paths[l_idx][0]
        criticality = get_tank_criticality(first_tank)
        dip_time = float(first_tank.get("dip_time_sec", 0))
        tank_complete_at = (l_idx * 60.0) + dip_time
        load_states.append(
            {
                "next_hop": 0,
                "ready_at": 0.0 + (l_idx * 60.0),
                "tank_complete_at": tank_complete_at,
                "current_stn": int(first_tank.get("station_no", 0)),
                "criticality": criticality,
                "waiting_since": None if criticality == "HIGH" else (l_idx * 60.0),
                "finished": False,
            }
        )

    wagon_times = {w_id: 0.0 for w_id in config}
    # map Station ID -> distance_mm
    stn_dist = {
        int(t.get("station_no", 0)): float(t.get("distance_mm", 0)) for t in tanks
    }

    wagon_pos_mm = {
        w_id: stn_dist.get(config[w_id]["basic_pos"], 0.0) for w_id in config
    }
    wagon_stn = {w_id: config[w_id]["basic_pos"] for w_id in config}
    step_counters = {w_id: 0 for w_id in config}

    station_free_at = defaultdict(float)

    # --- 3. Reactive Scheduler Loop ---
    total_hops_to_gen = sum(len(p) - 1 for p in load_paths)
    hops_generated = 0
    max_iter = total_hops_to_gen * 20
    iterations = 0

    while hops_generated < total_hops_to_gen and iterations < max_iter:
        iterations += 1
        best_task = None
        min_start_time = float("inf")

        for l_idx in range(num_loads):
            state = load_states[l_idx]
            if state["finished"] or state["next_hop"] >= len(load_paths[l_idx]) - 1:
                continue

            hop_idx = state["next_hop"]
            curr_tank = load_paths[l_idx][hop_idx]
            next_tank = load_paths[l_idx][hop_idx + 1]

            s_from = int(curr_tank.get("station_no", 0))
            s_to = int(next_tank.get("station_no", 0))
            d_from = float(curr_tank.get("distance_mm", 0))
            d_to = float(next_tank.get("distance_mm", 0))
            t_row = int(curr_tank.get("Row") or curr_tank.get("row") or 1)

            wagon_id = None
            for w_name, w_cfg in config.items():
                if (
                    int(w_cfg.get("row", 1)) == t_row
                    and w_cfg["min_stn"] <= s_from <= w_cfg["max_stn"]
                ):
                    wagon_id = w_name
                    # Note: We pick the first wagon that CAN do this.
                    # In a row with 3 wagons, this might lead to some idling.
                    break

            if not wagon_id:
                continue

            # Times
            wagon_ready = wagon_times[wagon_id]
            stn_from_ready = state["ready_at"]
            stn_to_free = station_free_at[s_to]

            w_cfg = config[wagon_id]

            # 1. Earliest the load can be picked up
            earliest_pickup = max(wagon_ready, stn_from_ready)

            # 2. Add travel to s_from if not already there
            travel_to_f = 0
            if wagon_stn[wagon_id] != s_from:
                dist = abs(d_from - wagon_pos_mm[wagon_id])
                travel_to_f = calculate_time_value(
                    dist, max(0, dist - 500), 500, w_cfg["sf"], w_cfg["f"], w_cfg["s"]
                )

            pickup_start = earliest_pickup + travel_to_f

            # Check Collision for travel to s_from
            wait_for_collision = 0
            if enable_collision_engine:
                # Logic check: is path to d_from clear of other wagons at 'pickup_start'?
                # For simplicity in this logic, we check if any other wagon on same row
                # is between current_pos and d_from within safe_distance.
                for other_w, other_pos in wagon_pos_mm.items():
                    if other_w == wagon_id or other_w not in config:
                        continue
                    if config[other_w]["row"] != t_row:
                        continue
                    # If the other wagon is "blocking" or too close
                    if abs(d_from - other_pos) < safe_distance_mm:
                        # We wait for that wagon's time (greedy)
                        wait_for_collision = max(
                            wait_for_collision, wagon_times[other_w] - pickup_start
                        )

            pickup_start += max(0, wait_for_collision)

            # Destination arrival
            dist_move = abs(d_to - d_from)
            tt = calculate_time_value(
                dist_move,
                max(0, dist_move - 500),
                500,
                w_cfg["sf"],
                w_cfg["f"],
                w_cfg["s"],
            )

            arrival_to_dist = pickup_start + w_cfg["lift_time"] + tt

            # Collision for travel to s_to
            wait_for_to = 0
            if enable_collision_engine:
                for other_w, other_pos in wagon_pos_mm.items():
                    if other_w == wagon_id or other_w not in config:
                        continue
                    if config[other_w]["row"] != t_row:
                        continue
                    if abs(d_to - other_pos) < safe_distance_mm:
                        wait_for_to = max(
                            wait_for_to, wagon_times[other_w] - arrival_to_dist
                        )

            arrival_to_dist += max(0, wait_for_to)

            # Check destination tank availability
            # In sequential mode with criticality, also check if tank processing is complete
            tank_available = True
            if processing_mode == "sequential":
                criticality = state["criticality"]
                tank_complete_at = state.get("tank_complete_at", 0.0)

                if criticality == "HIGH":
                    if arrival_to_dist < tank_complete_at:
                        tank_available = False
                elif criticality == "LOW":
                    max_hold = state.get("waiting_since")
                    if max_hold is not None:
                        if arrival_to_dist > tank_complete_at + MAX_HOLD_TIME_LOW:
                            pass
                        elif arrival_to_dist < tank_complete_at:
                            tank_available = False
                    else:
                        if arrival_to_dist < tank_complete_at:
                            tank_available = False

            if arrival_to_dist >= stn_to_free and tank_available:
                if pickup_start < min_start_time:
                    min_start_time = pickup_start
                    best_task = {
                        "load_idx": l_idx,
                        "wagon_id": wagon_id,
                        "s_from": s_from,
                        "s_to": s_to,
                        "d_from": d_from,
                        "d_to": d_to,
                        "start_time": pickup_start,
                        "travel_time": tt,
                        "wait_col": max(0, wait_for_collision + wait_for_to),
                        "t_row": t_row,
                        "next_tank": next_tank,
                    }

        if not best_task:
            break

        # --- 4. Execute Task ---
        l_idx = best_task["load_idx"]
        w_id = best_task["wagon_id"]
        s_f, s_t = best_task["s_from"], best_task["s_to"]
        d_f, d_t = best_task["d_from"], best_task["d_to"]
        acc = best_task["start_time"]
        w_cfg = config[w_id]

        # Insert WAIT if collision was detected
        if best_task["wait_col"] > 0:
            step_counters[w_id] += 1
            sequence_data.append(
                [
                    best_task["t_row"],
                    w_id,
                    step_counters[w_id],
                    f"WAIT_INTERLOCK(COLLISION)",
                    s_f,
                    "0.00",
                    f"{acc:.2f}",
                    f"L{l_idx}",
                    f"{best_task['wait_col']:.2f}",
                ]
            )
            acc += best_task["wait_col"]

        # GET FROM
        step_counters[w_id] += 1
        sequence_data.append(
            [
                best_task["t_row"],
                w_id,
                step_counters[w_id],
                "GET FROM",
                s_f,
                f"{w_cfg['lift_time']:.2f}",
                f"{acc:.2f}",
                f"L{l_idx}",
                "0.00",
            ]
        )
        acc += w_cfg["lift_time"]

        # Release source
        station_free_at[s_f] = 0

        # PUT ON
        tt = best_task["travel_time"]
        step_counters[w_id] += 1
        sequence_data.append(
            [
                best_task["t_row"],
                w_id,
                step_counters[w_id],
                "PUT ON",
                s_t,
                f"{(tt + w_cfg['lower_time']):.2f}",
                f"{acc:.2f}",
                f"L{l_idx}",
                "0.00",
            ]
        )
        acc += tt + w_cfg["lower_time"]

        # Update State
        dip = float(best_task["next_tank"].get("dip_time_sec", 0))
        station_free_at[s_t] = acc + dip

        wagon_times[w_id] = acc
        wagon_pos_mm[w_id] = d_t
        wagon_stn[w_id] = s_t

        load_states[l_idx]["next_hop"] += 1
        load_states[l_idx]["ready_at"] = acc + dip
        load_states[l_idx]["current_stn"] = s_t

        if processing_mode == "sequential" and state["next_hop"] + 1 < len(
            load_paths[l_idx]
        ):
            next_tank_obj = load_paths[l_idx][state["next_hop"] + 1]
            load_states[l_idx]["criticality"] = get_tank_criticality(next_tank_obj)
            load_states[l_idx]["tank_complete_at"] = acc + dip
            if load_states[l_idx]["criticality"] == "LOW":
                load_states[l_idx]["waiting_since"] = acc + dip
            else:
                load_states[l_idx]["waiting_since"] = None

        if load_states[l_idx]["next_hop"] >= len(load_paths[l_idx]) - 1:
            step_counters[w_id] += 1
            sequence_data.append(
                [
                    best_task["t_row"],
                    w_id,
                    step_counters[w_id],
                    "GET FROM",
                    s_t,
                    f"{w_cfg['lift_time']:.2f}",
                    f"{acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            wagon_times[w_id] = acc + w_cfg["lift_time"]
            load_states[l_idx]["finished"] = True
            station_free_at[s_t] = 0

        hops_generated += 1

    headers = sequence_data[0]
    rows = sequence_data[1:]
    rows.sort(key=lambda x: float(x[headers.index("AccumulatedTime")]))
    return [headers] + rows


def gap_analysis_grouped_sequence(
    tanks, config, enable_collision_engine=True, safe_distance_mm=500.0, num_loads=1
):
    """
    Grouped routing scheduler: Multiple source tanks feed into collector tanks.

    Expected flow:
    - 1 → 2, 1 → 3, 1 → 4 (Loading → A acid)
    - 2 → 5, 3 → 5, 4 → 5 (A acid → Post Rinse)
    - 5 → 6 (Post Rinse → B acid)
    - 6 → 9, 7 → 9, 8 → 9 (B acid → Rinse)
    - 9 → 10 (Rinse → Unloading)
    """
    if safe_distance_mm is None:
        safe_distance_mm = 500.0

    headers = [
        "Row",
        "Wagon",
        "Step No",
        "Command",
        "Value",
        "TravelTime",
        "AccumulatedTime",
        "CollisionFlag",
        "WaitTime",
    ]
    sequence_data = [headers]

    tanks_sorted = sorted(tanks, key=lambda x: int(x.get("station_no", 0)))

    process_groups = []
    current_group = []
    current_process = None

    for tank in tanks_sorted:
        proc = (tank.get("process_name") or "").strip().lower()
        if current_process is None:
            current_process = proc
            current_group = [tank]
        elif proc == current_process and proc != "":
            current_group.append(tank)
        else:
            if current_group:
                process_groups.append(current_group)
            current_process = proc
            current_group = [tank]

    if current_group:
        process_groups.append(current_group)

    tank_dict = {int(t.get("station_no", 0)): t for t in tanks}

    # Build pipeline stages - each stage is (from_stations, to_stations)
    def build_pipeline_stages(process_groups):
        stages = []
        for i in range(len(process_groups)):
            current_group = process_groups[i]
            current_stations = [int(t.get("station_no", 0)) for t in current_group]

            if i < len(process_groups) - 1:
                next_group = process_groups[i + 1]
                next_stations = [int(t.get("station_no", 0)) for t in next_group]

                if len(current_stations) == 1 and len(next_stations) > 1:
                    src = current_stations[0]
                    stages.append(([src], next_stations))
                elif len(current_stations) > 1 and len(next_stations) == 1:
                    dst = next_stations[0]
                    stages.append((current_stations, [dst]))
                elif len(current_stations) > 1 and len(next_stations) > 1:
                    dst = next_stations[0]
                    stages.append((current_stations, [dst]))
                else:
                    stages.append((current_stations, next_stations))

        return stages

    pipeline_stages = build_pipeline_stages(process_groups)

    stn_dist = {
        int(t.get("station_no", 0)): float(t.get("distance_mm", 0)) for t in tanks
    }
    wagon_pos_mm = {
        w_id: stn_dist.get(config[w_id]["basic_pos"], 0.0) for w_id in config
    }
    wagon_stn = {w_id: config[w_id]["basic_pos"] for w_id in config}
    step_counters = {w_id: 0 for w_id in config}

    # Initialize items with their complete path through the pipeline
    # Step 1: First move all items from station 1 to their destinations (2,3,4)
    # Step 2: Then process each item through the full pipeline

    # Build complete paths for each item (only the pipeline part after stage 1)
    all_paths = []
    if len(pipeline_stages) > 0:
        first_stage = pipeline_stages[0]
        _, first_destinations = first_stage
        for dest in first_destinations:
            # Build complete path for this item (stages 1-4 only, excluding stage 0)
            path = []
            current_source = dest  # Start at destination of stage 0

            # Stage 1->2: from destination (2,3,4) to station 5
            path.append((current_source, 5))

            # Stage 2->3: from station 5 to (6, 7, or 8)
            to_stn = dest + 4
            path.append((5, to_stn))
            current_source = to_stn

            # Stage 3->4: from (6,7,8) to station 9
            path.append((current_source, 9))

            # Stage 4->5: from station 9 to station 10
            path.append((9, 10))

            all_paths.append(path)

    # Process each item ONE BY ONE through ALL stages before moving to next item
    for l_idx in range(num_loads):
        start_time = l_idx * 60.0
        current_acc = start_time

        # Get the first destinations (2,3,4) - we need to process each separately
        first_stage = pipeline_stages[0]
        from_stations, to_stations = first_stage  # [1], [2,3,4]

        # For each destination from station 1, process the complete pipeline
        for dest_idx, dest in enumerate(to_stations):
            # Stage 0: 1 -> dest (2, 3, or 4)
            from_stn = from_stations[0]  # 1
            to_stn = dest

            from_tank = tank_dict.get(from_stn, {})
            to_tank = tank_dict.get(to_stn, {})

            from_dist = stn_dist.get(from_stn, 0)
            to_dist = stn_dist.get(to_stn, 0)
            from_row = int(from_tank.get("Row") or from_tank.get("row") or 1)
            to_row = int(to_tank.get("Row") or to_tank.get("row") or 1)

            wagon_id = None
            for w_name, w_cfg in config.items():
                if int(w_cfg.get("row", 1)) == to_row:
                    wagon_id = w_name
                    break
            if not wagon_id:
                wagon_id = list(config.keys())[0] if config else None
            if not wagon_id:
                continue
            w_cfg = config[wagon_id]

            travel_to_from = 0
            if wagon_stn[wagon_id] != from_stn:
                dist = abs(from_dist - wagon_pos_mm[wagon_id])
                travel_to_from = calculate_time_value(
                    dist, max(0, dist - 500), 500, w_cfg["sf"], w_cfg["f"], w_cfg["s"]
                )

            current_acc += travel_to_from

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    from_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "GET FROM",
                    from_stn,
                    f"{w_cfg['lift_time']:.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += w_cfg["lift_time"]

            travel_to_to = abs(to_dist - from_dist)
            tt = calculate_time_value(
                travel_to_to,
                max(0, travel_to_to - 500),
                500,
                w_cfg["sf"],
                w_cfg["f"],
                w_cfg["s"],
            )

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    to_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "PUT ON",
                    to_stn,
                    f"{(tt + w_cfg['lower_time']):.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += tt + w_cfg["lower_time"]

            dip = float(to_tank.get("dip_time_sec", 0))
            current_acc += dip

            wagon_pos_mm[wagon_id] = to_dist
            wagon_stn[wagon_id] = to_stn

            # NOW continue with remaining stages for THIS item
            # Stage 1: from dest (2/3/4) -> 5
            from_stn = to_stn  # 2, 3, or 4
            to_stn = 5

            from_tank = tank_dict.get(from_stn, {})
            to_tank = tank_dict.get(to_stn, {})

            from_dist = stn_dist.get(from_stn, 0)
            to_dist = stn_dist.get(to_stn, 0)
            from_row = int(from_tank.get("Row") or from_tank.get("row") or 1)
            to_row = int(to_tank.get("Row") or to_tank.get("row") or 1)

            travel_to_from = 0
            if wagon_stn[wagon_id] != from_stn:
                dist = abs(from_dist - wagon_pos_mm[wagon_id])
                travel_to_from = calculate_time_value(
                    dist, max(0, dist - 500), 500, w_cfg["sf"], w_cfg["f"], w_cfg["s"]
                )

            current_acc += travel_to_from

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    from_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "GET FROM",
                    from_stn,
                    f"{w_cfg['lift_time']:.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += w_cfg["lift_time"]

            travel_to_to = abs(to_dist - from_dist)
            tt = calculate_time_value(
                travel_to_to,
                max(0, travel_to_to - 500),
                500,
                w_cfg["sf"],
                w_cfg["f"],
                w_cfg["s"],
            )

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    to_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "PUT ON",
                    to_stn,
                    f"{(tt + w_cfg['lower_time']):.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += tt + w_cfg["lower_time"]

            dip = float(to_tank.get("dip_time_sec", 0))
            current_acc += dip

            wagon_pos_mm[wagon_id] = to_dist
            wagon_stn[wagon_id] = to_stn

            # Stage 2: from 5 -> 6/7/8 (based on original dest)
            from_stn = 5
            to_stn = dest + 4  # 2+4=6, 3+4=7, 4+4=8

            from_tank = tank_dict.get(from_stn, {})
            to_tank = tank_dict.get(to_stn, {})

            from_dist = stn_dist.get(from_stn, 0)
            to_dist = stn_dist.get(to_stn, 0)
            from_row = int(from_tank.get("Row") or from_tank.get("row") or 1)
            to_row = int(to_tank.get("Row") or to_tank.get("row") or 1)

            travel_to_from = 0
            if wagon_stn[wagon_id] != from_stn:
                dist = abs(from_dist - wagon_pos_mm[wagon_id])
                travel_to_from = calculate_time_value(
                    dist, max(0, dist - 500), 500, w_cfg["sf"], w_cfg["f"], w_cfg["s"]
                )

            current_acc += travel_to_from

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    from_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "GET FROM",
                    from_stn,
                    f"{w_cfg['lift_time']:.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += w_cfg["lift_time"]

            travel_to_to = abs(to_dist - from_dist)
            tt = calculate_time_value(
                travel_to_to,
                max(0, travel_to_to - 500),
                500,
                w_cfg["sf"],
                w_cfg["f"],
                w_cfg["s"],
            )

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    to_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "PUT ON",
                    to_stn,
                    f"{(tt + w_cfg['lower_time']):.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += tt + w_cfg["lower_time"]

            dip = float(to_tank.get("dip_time_sec", 0))
            current_acc += dip

            wagon_pos_mm[wagon_id] = to_dist
            wagon_stn[wagon_id] = to_stn

            # Stage 3: from 6/7/8 -> 9
            from_stn = to_stn
            to_stn = 9

            from_tank = tank_dict.get(from_stn, {})
            to_tank = tank_dict.get(to_stn, {})

            from_dist = stn_dist.get(from_stn, 0)
            to_dist = stn_dist.get(to_stn, 0)
            from_row = int(from_tank.get("Row") or from_tank.get("row") or 1)
            to_row = int(to_tank.get("Row") or to_tank.get("row") or 1)

            travel_to_from = 0
            if wagon_stn[wagon_id] != from_stn:
                dist = abs(from_dist - wagon_pos_mm[wagon_id])
                travel_to_from = calculate_time_value(
                    dist, max(0, dist - 500), 500, w_cfg["sf"], w_cfg["f"], w_cfg["s"]
                )

            current_acc += travel_to_from

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    from_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "GET FROM",
                    from_stn,
                    f"{w_cfg['lift_time']:.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += w_cfg["lift_time"]

            travel_to_to = abs(to_dist - from_dist)
            tt = calculate_time_value(
                travel_to_to,
                max(0, travel_to_to - 500),
                500,
                w_cfg["sf"],
                w_cfg["f"],
                w_cfg["s"],
            )

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    to_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "PUT ON",
                    to_stn,
                    f"{(tt + w_cfg['lower_time']):.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += tt + w_cfg["lower_time"]

            dip = float(to_tank.get("dip_time_sec", 0))
            current_acc += dip

            wagon_pos_mm[wagon_id] = to_dist
            wagon_stn[wagon_id] = to_stn

            # Stage 4: from 9 -> 10
            from_stn = 9
            to_stn = 10

            from_tank = tank_dict.get(from_stn, {})
            to_tank = tank_dict.get(to_stn, {})

            from_dist = stn_dist.get(from_stn, 0)
            to_dist = stn_dist.get(to_stn, 0)
            from_row = int(from_tank.get("Row") or from_tank.get("row") or 1)
            to_row = int(to_tank.get("Row") or to_tank.get("row") or 1)

            travel_to_from = 0
            if wagon_stn[wagon_id] != from_stn:
                dist = abs(from_dist - wagon_pos_mm[wagon_id])
                travel_to_from = calculate_time_value(
                    dist, max(0, dist - 500), 500, w_cfg["sf"], w_cfg["f"], w_cfg["s"]
                )

            current_acc += travel_to_from

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    from_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "GET FROM",
                    from_stn,
                    f"{w_cfg['lift_time']:.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += w_cfg["lift_time"]

            travel_to_to = abs(to_dist - from_dist)
            tt = calculate_time_value(
                travel_to_to,
                max(0, travel_to_to - 500),
                500,
                w_cfg["sf"],
                w_cfg["f"],
                w_cfg["s"],
            )

            step_counters[wagon_id] += 1
            sequence_data.append(
                [
                    to_row,
                    wagon_id,
                    step_counters[wagon_id],
                    "PUT ON",
                    to_stn,
                    f"{(tt + w_cfg['lower_time']):.2f}",
                    f"{current_acc:.2f}",
                    f"L{l_idx}",
                    "0.00",
                ]
            )
            current_acc += tt + w_cfg["lower_time"]

            dip = float(to_tank.get("dip_time_sec", 0))
            current_acc += dip

            wagon_pos_mm[wagon_id] = to_dist
            wagon_stn[wagon_id] = to_stn

        # Step 2: Now process remaining stages - process each stage for ALL items before moving to next
        # For each stage (1 to 4), process all items
        for stage_offset in range(len(all_paths[0]) if all_paths else 0):
            for path in all_paths:
                if stage_offset >= len(path):
                    continue
                from_stn, to_stn = path[stage_offset]

                from_tank = tank_dict.get(from_stn, {})
                to_tank = tank_dict.get(to_stn, {})

                from_dist = stn_dist.get(from_stn, 0)
                to_dist = stn_dist.get(to_stn, 0)
                from_row = int(from_tank.get("Row") or from_tank.get("row") or 1)
                to_row = int(to_tank.get("Row") or to_tank.get("row") or 1)

                wagon_id = None
                for w_name, w_cfg in config.items():
                    if int(w_cfg.get("row", 1)) == to_row:
                        wagon_id = w_name
                        break
                if not wagon_id:
                    wagon_id = list(config.keys())[0] if config else None
                if not wagon_id:
                    continue
                w_cfg = config[wagon_id]

                travel_to_from = 0
                if wagon_stn[wagon_id] != from_stn:
                    dist = abs(from_dist - wagon_pos_mm[wagon_id])
                    travel_to_from = calculate_time_value(
                        dist,
                        max(0, dist - 500),
                        500,
                        w_cfg["sf"],
                        w_cfg["f"],
                        w_cfg["s"],
                    )

                current_acc += travel_to_from

                step_counters[wagon_id] += 1
                sequence_data.append(
                    [
                        from_row,
                        wagon_id,
                        step_counters[wagon_id],
                        "GET FROM",
                        from_stn,
                        f"{w_cfg['lift_time']:.2f}",
                        f"{current_acc:.2f}",
                        f"L{l_idx}",
                        "0.00",
                    ]
                )
                current_acc += w_cfg["lift_time"]

                travel_to_to = abs(to_dist - from_dist)
                tt = calculate_time_value(
                    travel_to_to,
                    max(0, travel_to_to - 500),
                    500,
                    w_cfg["sf"],
                    w_cfg["f"],
                    w_cfg["s"],
                )

                step_counters[wagon_id] += 1
                sequence_data.append(
                    [
                        to_row,
                        wagon_id,
                        step_counters[wagon_id],
                        "PUT ON",
                        to_stn,
                        f"{(tt + w_cfg['lower_time']):.2f}",
                        f"{current_acc:.2f}",
                        f"L{l_idx}",
                        "0.00",
                    ]
                )
                current_acc += tt + w_cfg["lower_time"]

                dip = float(to_tank.get("dip_time_sec", 0))
                current_acc += dip

                wagon_pos_mm[wagon_id] = to_dist
                wagon_stn[wagon_id] = to_stn

    headers = sequence_data[0]
    rows = sequence_data[1:]
    rows.sort(key=lambda x: float(x[headers.index("AccumulatedTime")]))
    return [headers] + rows


def generate_sequence_from_data(
    tanks_data, config=None, verbose=True, num_loads=1, processing_mode="sequential"
):
    """
    Generate sequence from tanks data directly (list of dicts or pandas DataFrame).
    This function accepts data directly from GUI to avoid CSV file collisions.

    Args:
        tanks_data: List of dictionaries or pandas DataFrame with tank data
        config: Optional wagon config dict. If None, loads from default location.
        verbose: If True, prints progress messages
        num_loads: int - number of loads to schedule concurrently
        processing_mode: "sequential" for sequential processing with criticality,
                        "load_balance" for original load rotation logic

    Returns:
        List of lists representing the sequence [headers, row1, row2, ...]
    """
    try:
        # Convert DataFrame to list of dicts if needed
        try:
            # Try pandas DataFrame conversion
            if hasattr(tanks_data, "to_dict") and hasattr(tanks_data, "columns"):
                # It's a pandas DataFrame
                tanks = tanks_data.to_dict("records")
            elif isinstance(tanks_data, list):
                # Already a list of dicts
                tanks = tanks_data
            else:
                raise ValueError(
                    "tanks_data must be a list of dicts or pandas DataFrame"
                )
        except AttributeError:
            # Fallback: if it's iterable, try to convert
            if isinstance(tanks_data, list):
                tanks = tanks_data
            else:
                raise ValueError(
                    "tanks_data must be a list of dicts or pandas DataFrame"
                )

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
            print(
                f"\nGenerating Multi-Load Sequence for {num_loads} loads from {len(tanks)} tanks..."
            )
            print(f"Processing mode: {processing_mode}")

        if processing_mode == "grouped":
            unified_seq = gap_analysis_grouped_sequence(
                tanks, config, num_loads=num_loads
            )
        else:
            unified_seq = gap_analysis_sequence(
                tanks, config, num_loads=num_loads, processing_mode=processing_mode
            )

        if verbose:
            if len(unified_seq) > 1:
                print(
                    tabulate(unified_seq[1:], headers=unified_seq[0], tablefmt="grid")
                )
            else:
                print(
                    "\nNo moves could be assigned to any wagon based on their station ranges."
                )

        return unified_seq

    except Exception as e:
        if verbose:
            print(f"Error generating sequence: {e}")
            import traceback

            traceback.print_exc()
        raise


def generate_sequence_from_tanks(
    csv_path=None,
    tanks_data=None,
    speeds_input=None,
    config=None,
    num_loads=1,
    processing_mode="sequential",
):
    """
    Generate sequence from CSV file or direct data input.
    Supports both CSV file path and direct data input for GUI integration.

    Args:
        csv_path: Path to CSV file (optional if tanks_data is provided)
        tanks_data: Direct data input - list of dicts or DataFrame (optional if csv_path is provided)
        speeds_input: Deprecated, kept for backward compatibility
        config: Optional wagon config dict. If None, loads from default location.
        num_loads: int - number of loads for concurrent scheduling
        processing_mode: "sequential" for sequential processing with criticality,
                        "load_balance" for original load rotation logic

    Returns:
        List of lists representing the sequence [headers, row1, row2, ...]
    """
    # If tanks_data is provided, use direct data input
    if tanks_data is not None:
        return generate_sequence_from_data(
            tanks_data,
            config=config,
            verbose=True,
            num_loads=num_loads,
            processing_mode=processing_mode,
        )

    # Otherwise, use CSV file (backward compatibility)
    if csv_path is None:
        csv_path = tanks_csv_default

    print(f"Processing sequence for {csv_path} ({num_loads} loads)...")
    print(f"Processing mode: {processing_mode}")
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
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            tanks = list(reader)

        # Generate Sequence
        return generate_sequence_from_data(
            tanks, config=config, verbose=True, processing_mode=processing_mode
        )

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
        with open(csv_path, "r", encoding="utf-8") as f:
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
    sequence_data = [
        ["Row", "Wagon", "Step No", "Command", "Value", "TravelTime", "AccumulatedTime"]
    ]

    # Simplified AI Step logic
    for t in range(len(tanks) * 2):  # Heuristic: 2 steps per tank (GET/PUT)
        a_cmd, a_stn = agent.select_action(curr_cmd_seq, curr_stn_seq, state)
        next_state, reward, done = env.step(a_cmd, a_stn)

        cmd_name = inv_vocab.get(a_cmd, "UNKNOWN")

        # Identify wagon for display
        wagon_id = "AI_Wagon"
        for w_name, w_cfg in config.items():
            if w_cfg["min_stn"] <= a_stn <= w_cfg["max_stn"]:
                wagon_id = w_name
                break

        # Dummy times for AI mode log (inference doesn't have physics integrated yet)
        sequence_data.append([1, wagon_id, t + 1, cmd_name, a_stn, "0.00", "0.00"])

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
def calculate_actual_dip_times(tanks, sequence_data):
    """
    Analytics Engine: Calculates actual dip times from a generated sequence.
    Derived from industrial_diptime_system.py logic.
    """
    if not sequence_data or len(sequence_data) < 2:
        return []

    headers = sequence_data[0]
    rows = sequence_data[1:]

    stations_meta = {}
    for t in tanks:
        stn_id = str(t.get("station_no", ""))
        if stn_id:
            stations_meta[stn_id] = {
                "Process Name": t.get("process_name", "Unknown"),
                "Distance": t.get("distance_mm", 0),
                "Target Dip": float(t.get("dip_time_sec", 0)),
            }

    stn_col = headers.index("Value")
    cmd_col = headers.index("Command")
    acc_col = headers.index("AccumulatedTime")
    wagon_col = headers.index("Wagon")
    load_col = headers.index("CollisionFlag") if "CollisionFlag" in headers else None

    stn_events = defaultdict(list)
    for row in rows:
        cmd = str(row[cmd_col]).upper()
        stn = str(row[stn_col])
        time_val = float(row[acc_col])
        wagon = str(row[wagon_col])
        load_id = str(row[load_col]) if load_col is not None else "L0"

        if "PUT ON" in cmd:
            stn_events[stn].append(
                {"type": "IN", "load": load_id, "time": time_val, "wagon": wagon}
            )
        elif "GET FROM" in cmd:
            stn_events[stn].append(
                {"type": "OUT", "load": load_id, "time": time_val, "wagon": wagon}
            )

    dip_results = []
    for stn_id, events in stn_events.items():
        events.sort(key=lambda x: x["time"])
        ins = [e for e in events if e["type"] == "IN"]
        outs = [e for e in events if e["type"] == "OUT"]
        used_outs = set()

        for in_act in ins:
            matched = False
            for j, out_act in enumerate(outs):
                if (
                    j not in used_outs
                    and out_act["load"] == in_act["load"]
                    and out_act["time"] >= in_act["time"]
                ):
                    used_outs.add(j)
                    dip_results.append(
                        {
                            "Station": stn_id,
                            "Load": in_act["load"],
                            "In Time": in_act["time"],
                            "Out Time": out_act["time"],
                            "Duration": out_act["time"] - in_act["time"],
                            "Wagon In": in_act["wagon"],
                            "Wagon Out": out_act["wagon"],
                        }
                    )
                    matched = True
                    break

            if not matched:
                for j, out_act in enumerate(outs):
                    if j not in used_outs and out_act["time"] >= in_act["time"]:
                        used_outs.add(j)
                        dip_results.append(
                            {
                                "Station": stn_id,
                                "Load": in_act["load"],
                                "In Time": in_act["time"],
                                "Out Time": out_act["time"],
                                "Duration": out_act["time"] - in_act["time"],
                                "Wagon In": in_act["wagon"],
                                "Wagon Out": out_act["wagon"],
                            }
                        )
                        matched = True
                        break

    final_output = []
    for res in dip_results:
        meta = stations_meta.get(res["Station"], {})
        final_output.append(
            {
                "Station": res["Station"],
                "Process": meta.get("Process Name", "N/A"),
                "Load": res["Load"],
                "Arrival": f"{res['In Time']:.2f}",
                "Departure": f"{res['Out Time']:.2f}",
                "Actual Dip": f"{res['Duration']:.2f}s",
                "Target Dip": f"{meta.get('Target Dip', 0):.2f}s",
                "Variance": f"{(res['Duration'] - meta.get('Target Dip', 0)):.2f}s",
                "Wagons": f"{res['Wagon In']} -> {res['Wagon Out']}",
            }
        )
    final_output.sort(key=lambda x: float(x["Arrival"]))
    return final_output


def main():
    global wagon_config_csv, row_config_csv
    parser = argparse.ArgumentParser(
        description="Sequence Generation and AI Training Tool"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gen", "train", "ai_gen"],
        default="gen",
        help="Execution mode",
    )
    parser.add_argument(
        "--input", type=str, help="Path to tanks CSV", default=tanks_csv_default
    )
    parser.add_argument(
        "--speeds",
        type=str,
        help="Wagon Name (Optional, now uses config ranges)",
        default="Wagon 1",
    )
    parser.add_argument(
        "--config", type=str, help="Path to wagon config CSV", default=wagon_config_csv
    )
    parser.add_argument(
        "--row_config", type=str, help="Path to row config CSV", default=row_config_csv
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model (.pth)",
        default=os.path.join(parent_dir, "model_v9.pth"),
    )
    parser.add_argument(
        "--load", type=int, help="Load Number for tank rotation", default=0
    )
    parser.add_argument(
        "--processing_mode",
        type=str,
        choices=["sequential", "grouped", "load_balance"],
        default="grouped",
        help="Processing mode for sequence generation",
    )

    args = parser.parse_args()
    wagon_config_csv = args.config
    row_config_csv = args.row_config
    num_loads = args.load if args.load > 0 else 1

    if args.mode == "gen":
        generate_sequence_from_tanks(
            csv_path=args.input,
            speeds_input=args.speeds,
            config=load_wagon_config(wagon_config_csv),
            num_loads=num_loads,
            processing_mode=args.processing_mode,
        )
    elif args.mode == "ai_gen":
        generate_sequence_ai(
            csv_path=args.input,
            model_path=args.model,
            config=load_wagon_config(wagon_config_csv),
        )
    else:
        train_data, adj_list, vocab_cmd = load_data()
        if not train_data:
            print("No training data found. Exiting.")
            return
        agent = PPOAgent(vocab_cmd, MAX_STATIONS, adj_list)
        env = WagonEnv(adj_list, MAX_STATIONS)
        print("\nStarting RL Training Loop (1000 Episodes)...")
        for episode in range(1, 1001):
            state = env.reset()
            curr_cmd_seq, curr_stn_seq = [vocab_cmd["<SOS>"]], [0]
            ep_reward, collisions = 0, 0
            for t in range(MAX_STEPS):
                a_cmd, a_stn = agent.select_action(curr_cmd_seq, curr_stn_seq, state)
                next_state, reward, done = env.step(a_cmd, a_stn)
                agent.buffer.rewards.append(reward)
                agent.buffer.is_terminals.append(done)
                if reward < 0:
                    collisions += 1
                curr_cmd_seq.append(a_cmd)
                curr_stn_seq.append(a_stn)
                state = next_state
                ep_reward += reward
                if done:
                    break
            agent.update()
            if episode % 20 == 0:
                print(
                    f"Episode {episode}: Total Reward = {ep_reward} (Collisions/Invalid: {collisions})"
                )
        print("Training Finished. Model saved as 'model_v7.pth'")
        torch.save(agent.policy.state_dict(), "model_v7.pth")


if __name__ == "__main__":
    main()
