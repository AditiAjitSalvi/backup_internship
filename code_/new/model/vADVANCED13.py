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
                    'row': int(row.get('Row') or row.get('Row Number') or 1)
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

def gap_analysis_sequence(tanks, config):
    """
    Generates a unified sequence for all wagons in a single table.
    """
    sequence_data = [["Row", "Wagon", "Step No", "Command", "Value", "TravelTime", "AccumulatedTime"]]
    accumulated_times = {w_id: 0.0 for w_id in config}
    step_counters = {w_id: 0 for w_id in config}

    print(f"\n[INFO] Generating unified row-aware sequence for {len(config)} wagons...")

    for i in range(len(tanks) - 1):
        curr_tank = tanks[i]
        next_tank = tanks[i+1]
        
        try:
            s_curr = int(curr_tank.get('station_no', 0))
            s_next = int(next_tank.get('station_no', 0))
            t_row = int(curr_tank.get('Row') or curr_tank.get('row') or 1)
        except ValueError:
            continue

        # Identify wagon based on current station AND row
        wagon_id = None
        for w_name, w_cfg in config.items():
            w_row = int(w_cfg.get('row', 1))
            if w_row == t_row and w_cfg['min_stn'] <= s_curr <= w_cfg['max_stn']:
                wagon_id = w_name
                break
        
        if not wagon_id:
            continue

        w = config[wagon_id]
        dist_curr = float(curr_tank.get('distance_mm', 0))
        dist_next = float(next_tank.get('distance_mm', 0))
        dist_total = abs(dist_next - dist_curr)
        
        # Physics Parameters
        distance3 = 500.0 # Slow speed approach distance
        distance2 = max(0, dist_total - distance3)
        distance1 = dist_total 

        travel_time = calculate_time_value(distance1, distance2, distance3, w['sf'], w['f'], w['s'])
        acc_time = accumulated_times[wagon_id]
        
        # GET FROM
        step_counters[wagon_id] += 1
        sequence_data.append([
            t_row, wagon_id, step_counters[wagon_id], "GET FROM", s_curr, 
            f"{w['lift_time']:.2f}", f"{acc_time:.2f}"
        ])
        acc_time += w['lift_time']
        
        # PUT ON
        step_counters[wagon_id] += 1
        sequence_data.append([
            t_row, wagon_id, step_counters[wagon_id], "PUT ON", s_next, 
            f"{(travel_time + w['lower_time']):.2f}", f"{acc_time:.2f}"
        ])
        acc_time += travel_time + w['lower_time']
        
        accumulated_times[wagon_id] = acc_time
            
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
