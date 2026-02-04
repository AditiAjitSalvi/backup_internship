import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.distributions import Categorical
from collections import defaultdict

# ================================
# CONFIG
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"C:\Users\aditi\Downloads\Internship"
SEQUENCE_FILE = os.path.join(BASE_DIR, "sequnce for traing.csv")
SAFETY_FILE = os.path.join(DATA_DIR, "Internship", "WayTime-dB.mdb", "CrossTrolleyMaster.csv")
TANKS_FILE = os.path.join(BASE_DIR, "tanks_csv.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# 1. GNN: SAFETY MODULE
# ================================
class SafetyGNN(nn.Module):
    def __init__(self, num_stations, embed_dim=16):
        super().__init__()
        self.station_embed = nn.Embedding(num_stations + 1, embed_dim)
        self.conv1 = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()
    
    def forward(self, adj_matrix, active_nodes):
        # active_nodes: [batch, num_stations] (1 if wagon present)
        x = self.station_embed.weight # [num_nodes, dim]
        
        # Simple Logic: Propagate "Danger" logic
        # adj_matrix: [num_nodes, num_nodes] (1 if connected/interlock)
        out = torch.matmul(adj_matrix, x) 
        out = self.relu(self.conv1(out))
        
        return out # [num_nodes, dim] represents safety state of each node

    def get_safety_mask(self, current_node_embeddings, proposed_nodes):
        # Verify if proposed_nodes are safe given current state
        # Returns simple mask for now (1=Safe, 0=Unsafe)
        return torch.ones_like(proposed_nodes).float()

def load_safety_graph():
    try:
        df = pd.read_csv(SAFETY_FILE)
        # Build Adjacency Matrix for Interlocks
        # Nodes: Stations. Edge if they share CrossTrolley
        max_stn = max(df['Row1StationNo'].max(), df['Row2StationNo'].max()) + 1
        adj = torch.zeros((max_stn, max_stn))
        
        for _, row in df.iterrows():
            a, b = int(row['Row1StationNo']), int(row['Row2StationNo'])
            adj[a, b] = 1
            adj[b, a] = 1 # Undirected conflict
            
        print(f"✅ GNN: Built Safety Graph with {max_stn} nodes")
        return adj
    except Exception as e:
        print(f"⚠️ GNN Error: {e}")
        return torch.eye(MAX_STATIONS)

# ================================
# 2. TRANSFORMER: SEQUENCE MODEL
# ================================
class SequenceTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers=2
        )
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, src):
        # src: [seq_len, batch]
        x = self.embedding(src)
        x = self.transformer(x)
        return x[-1, :, :] # Return context of last step

def load_sequence_data():
    try:
        df = pd.read_csv(SEQUENCE_FILE)
        # Mapping Command+Station to Token ID
        # Simple Tokenizer: "Command_Station"
        df['token'] = df['command'] + "_" + df['station_no'].astype(str)
        tokens = df['token'].unique().tolist()
        vocab = {t: i for i, t in enumerate(tokens)}
        print(f"✅ Transformer: Loaded {len(tokens)} unique sequence tokens")
        return vocab, df
    except Exception as e:
        print(f"⚠️ Transformer Error: {e}")
        return {}, pd.DataFrame()

# ================================
# 3. PPO: OPTIMIZER
# ================================
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def get_action(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(state)

# ================================
# MAIN SCHEDULER
# ================================
MAX_STATIONS = 100 # Approx from data

def main():
    print("🚀 Initializing Hybrid Transformer-PPO-GNN Scheduler...")
    
    # 1. Load Data
    adj_matrix = load_safety_graph().to(device)
    vocab, seq_df = load_sequence_data()
    
    if not vocab:
        print("❌ Cannot proceed without sequence data.")
        return

    # 2. Init Models
    gnn = SafetyGNN(num_stations=MAX_STATIONS).to(device)
    transformer = SequenceTransformer(vocab_size=len(vocab)+1).to(device)
    ppo = PPOAgent(state_dim=32, action_dim=len(vocab)).to(device) # state_dim matches d_model
    
    optimizer = optim.Adam(list(gnn.parameters()) + list(transformer.parameters()) + list(ppo.parameters()), lr=0.001)
    
    # 3. Pseudo-Training Loop (Simulation)
    print("\n🧠 Starting Training loop...")
    for episode in range(5): # Short training for demo
        # Simulate a sequence state
        # In real scenario: Reset Env, get initial state
        
        # Fake input sequence (embedding IDs)
        dummy_seq = torch.randint(0, len(vocab), (10, 1)).to(device)
        
        # 1. Transformer encodes pattern
        context = transformer(dummy_seq)
        
        # 2. GNN encodes safety (dummy active nodes)
        # active = torch.zeros(1, MAX_STATIONS).to(device)
        # safety_embed = gnn(adj_matrix, active) 
        # For simplicity, we assume context includes safety info fused in a real full implementation
        
        # 3. PPO decides next move
        action, log_prob, value = ppo.get_action(context)
        
        # 4. Reward calculation (Check validity against sequences or rules)
        reward = 1.0 # Placeholder
        
        # 5. Update
        optimizer.zero_grad()
        loss = -log_prob * reward + (value - reward)**2 # Simple REINFORCE-like loss 
        loss.mean().backward()
        optimizer.step()
        
        print(f"   Episode {episode+1}: Loss = {loss.mean().item():.4f}")

    # 4. Generate Sequence
    print("\n✨ Generating Optimized Sequence...")
    # Map back tokens to CSV
    inv_vocab = {i: t for t, i in vocab.items()}
    output_rows = []
    
    # Generate 10 steps
    curr_seq = torch.randint(0, len(vocab), (5, 1)).to(device)
    for i in range(10):
        with torch.no_grad():
            ctx = transformer(curr_seq)
            action, _, _ = ppo.get_action(ctx)
            token_id = action.item()
            
            token_str = inv_vocab.get(token_id, "Unknown_0")
            cmd, stn = token_str.split('_')
            
            output_rows.append({
                'seq_id': 999, 'project_id': 1, 'wagon_no': 'Wagon 1', 
                'step_no': i+1, 'command': cmd, 'station_no': stn,
                'total_time_sec': (i+1)*60, 'valid': True
            })
            
            # Update sequence
            new_step = torch.tensor([[token_id]]).to(device)
            curr_seq = torch.cat([curr_seq[1:], new_step], dim=0)

    # Save
    out_df = pd.DataFrame(output_rows)
    out_path = os.path.join(BASE_DIR, "ml_sequence_output.csv")
    out_df.to_csv(out_path, index=False)
    print(f"💾 Sequence saved to {out_path}")
    print(out_df.head())

if __name__ == "__main__":
    main()
