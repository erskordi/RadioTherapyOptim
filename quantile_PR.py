import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from collections import deque
import matplotlib.pyplot as plt

# Import your optimized Numba Environment
from Numba_Test_Train import BeamAngleEnv

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# -----------------------------------------------------------
# 1. PRIORITIZED EXPERIENCE REPLAY (PER) UTILS (OPTIMIZED)
# -----------------------------------------------------------

class SumTree:
    """
    Optimized Binary SumTree (Iterative, No Recursion).
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        
        # Iterative propagation (Faster than recursion)
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            
            if left >= len(self.tree):
                break
            
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

        # SAFETY: Clamp index to valid range to prevent crashes due to float precision
        # (This replaces the need for the "if data == 0" check in your buffer)
        idx = min(idx, len(self.tree) - 1)
        
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        # New experiences get max priority to ensure they are seen at least once
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0: max_p = 1.0
        
        self.tree.add(max_p, (state, action, reward, next_state, done))

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        
        # Divide the range into batch_size segments
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Calculate IS Weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max() # Normalize

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Helper to stack dict obs
        def stack_obs(obs_list):
            return {
                "dose": np.stack([o["dose"] for o in obs_list]),
                "beams": np.stack([o["beams"] for o in obs_list]),
            }

        return (
            stack_obs(states),
            np.array(actions),
            np.array(rewards),
            stack_obs(next_states),
            np.array(dones),
            np.array(idxs),
            np.array(is_weight, dtype=np.float32)
        )

    def update_priorities(self, idxs, errors):
        for idx, err in zip(idxs, errors):
            p = (abs(err) + 1e-5) ** self.alpha
            self.tree.update(idx, p)
class DeepResNetEncoder(nn.Module):
    """
    A lighter 3-layer CNN (Faster than ResNet, better than shallow).
    """
    def __init__(self, input_channels=2):
        super().__init__()
        # Input: (2, 18, 18, 18)
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        # MaxPool reduces size by 2 each time: 18 -> 9 -> 4 -> 2
        self.pool = nn.MaxPool3d(2)
        
        # Calculate flat size: 
        # After pool1: 9x9x9
        # After pool2: 4x4x4
        # After pool3: 2x2x2
        self.flat_size = 128 * 2 * 2 * 2 # = 1024

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)

class EnsembleQuantileCritic(nn.Module):
    """
    TQC Critic: Outputs 'n_quantiles' values for each of 'ensemble_size' members.
    Total output shape: [Batch, Ensemble, Quantiles]
    """
    def __init__(self, obs_shape, beam_dim, action_dim, n_quantiles=25, ensemble_size=2):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_quantiles = n_quantiles
        
        self.encoder = DeepResNetEncoder()
        input_dim = self.encoder.flat_size + beam_dim + action_dim
        
        # Independent Critics
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, n_quantiles) # Output Values (not probs)
            ) for _ in range(ensemble_size)
        ])

    def forward(self, state, action):
        img_feat = self.encoder(state["dose"])
        beam_feat = state["beams"]
        x = torch.cat([img_feat, beam_feat, action], dim=1)
        
        # Stack outputs: [Batch, Ensemble, Quantiles]
        return torch.stack([net(x) for net in self.nets], dim=1)

class AdaptiveActor(nn.Module):
    """
    Actor with separate Geometry (Free) and Intensity (Limited) heads.
    """
    def __init__(self, obs_shape, beam_dim, action_dim, action_low, action_high):
        super().__init__()
        self.encoder = DeepResNetEncoder()
        input_dim = self.encoder.flat_size + beam_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim) 
        )
        
        self.register_buffer("low", torch.tensor(action_low))
        self.register_buffer("high", torch.tensor(action_high))
        
        # Adaptive Limit (Starts small, grows during training)
        self.current_limit = 0.1 

    def forward(self, state):
        img_feat = self.encoder(state["dose"])
        beam_feat = state["beams"]
        x = torch.cat([img_feat, beam_feat], dim=1)
        
        raw_action = self.net(x)
        
        # --- SPLIT ACTION ---
        # Assuming format: [Gantry, Couch, Energies(6), Intensities(L*N)]
        # Geometry + Energy part (Indices 0 to 7 approx)
        # Intensity part (Indices 8+)
        # We need to calculate split point based on dimensions
        total_dim = raw_action.shape[1]
        
        # Heuristic: The last chunk are intensities.
        # Based on your env: 1 Gantry + 1 Couch + 6 Energy + 6*16 Intensity = 104 dims?
        # Let's generalize: Intensity is usually the last 50-90% of the vector.
        # But for exactness, we apply Tanh to Geometry and Sigmoid * Limit to Intensities.
        # NOTE: For simplicity in this general implementation, we assume the 
        # FIRST 2 are Angles (Free), Next 6 are Energies (Free [0,1]), Rest are Intensities (Limited).
        
        geo_dim = 2 # Gantry, Couch
        energy_dim = 6
        int_dim = total_dim - geo_dim - energy_dim
        
        # 1. Angles (-1 to 1) -> Map to env bounds
        angles = torch.tanh(raw_action[:, :geo_dim])
        
        # 2. Energies (0 to 1) -> Sigmoid
        energies = torch.sigmoid(raw_action[:, geo_dim:geo_dim+energy_dim])
        
        # 3. Intensities (0 to Limit) -> Sigmoid * Limit
        intensities = torch.sigmoid(raw_action[:, geo_dim+energy_dim:]) * self.current_limit
        
        # Recombine
        combined = torch.cat([angles, energies, intensities], dim=1)
        
        # Scale to environment bounds
        # Angles need scaling from [-1, 1] to [low, high]
        # Energies and Intensities are already in [0, 1] relative scale (env expects absolute?)
        # Your env space is:
        # Angles: [-pi, pi] etc.
        # Energy/Int: [0, 1]
        
        # Global scaling for angles
        scaled = combined * 1.0 # placeholder clone
        
        # Map angles manually
        angle_low = self.low[:geo_dim]
        angle_high = self.high[:geo_dim]
        scaled[:, :geo_dim] = (angles * (angle_high - angle_low) / 2.0) + (angle_high + angle_low) / 2.0
        
        # Map others (assuming env bounds are 0-1 for others)
        # If env bounds are not 0-1, you need to scale here.
        # Based on your env code, energy/int bounds are 0-1. So direct output is fine.
        
        return scaled

# -----------------------------------------------------------
# 3. RACER AGENT (With TQC & PER)
# -----------------------------------------------------------

class RACER_TQC_PER:
    def __init__(self, env, config):
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha_cvar = 0.10  # Risk Sensitivity
        self.n_quantiles = 25
        self.batch_size = 64
        self.lr = 3e-4
        
        self.obs_shape = env.observation_space["dose"].shape
        self.beam_dim = env.observation_space["beams"].shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Networks
        self.actor = AdaptiveActor(self.obs_shape, self.beam_dim, self.action_dim, 
                                   env.action_space.low, env.action_space.high).to(device)
        self.critic = EnsembleQuantileCritic(self.obs_shape, self.beam_dim, self.action_dim, 
                                             self.n_quantiles).to(device)
        self.target_critic = EnsembleQuantileCritic(self.obs_shape, self.beam_dim, self.action_dim, 
                                                    self.n_quantiles).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        # Memory
        self.memory = PrioritizedReplayBuffer(50000)

    def get_action(self, state, explore=True):
        self.actor.eval()
        
        # Convert state to tensor
        dose = torch.FloatTensor(state["dose"]).to(device)
        beams = torch.FloatTensor(state["beams"]).to(device)
        
        # --- FIX IS HERE ---
        # Check for 4 dims (Channels, Depth, Height, Width)
        if dose.dim() == 4: 
            dose = dose.unsqueeze(0)   # Add batch dim -> (1, C, D, H, W)
            beams = beams.unsqueeze(0) # Add batch dim -> (1, Beam_Dim)
            
        state_tensor = {"dose": dose, "beams": beams}
            
        with torch.no_grad():
            action = self.actor(state_tensor)
            
        action = action.cpu().numpy()[0]
        
        if explore:
            noise = np.random.normal(0, 0.15, size=action.shape)
            action = np.clip(action + noise, env.action_space.low, env.action_space.high)
            
        return action

    def get_ensemble_cvar(self, state, action, alpha):
        """Pools atoms from ensemble, sorts, takes tail."""
        # [Batch, Ensemble, Quantiles]
        quantiles = self.critic(state, action)
        
        # Pool: [Batch, Ensemble * Quantiles]
        pooled = quantiles.view(quantiles.shape[0], -1)
        
        # Sort
        sorted_q, _ = torch.sort(pooled, dim=1)
        
        # Tail (Bottom alpha %)
        k = max(1, int(sorted_q.shape[1] * alpha))
        tail = sorted_q[:, :k]
        
        return tail.mean(dim=1, keepdim=True)

    def update(self, per_beta=0.4):
        if self.memory.tree.n_entries < self.batch_size:
            return None
            
        # 1. Sample from PER
        states, actions, rewards, next_states, dones, idxs, weights = self.memory.sample(self.batch_size, per_beta)
        
        # To Tensor
        def to_tensor(x): return torch.FloatTensor(x).to(device)
        
        s_dose = to_tensor(states["dose"])
        s_beams = to_tensor(states["beams"])
        state = {"dose": s_dose, "beams": s_beams}
        
        ns_dose = to_tensor(next_states["dose"])
        ns_beams = to_tensor(next_states["beams"])
        next_state = {"dose": ns_dose, "beams": ns_beams}
        
        act = to_tensor(actions)
        rew = to_tensor(rewards).unsqueeze(1)
        done = to_tensor(dones).unsqueeze(1)
        per_weights = to_tensor(weights).unsqueeze(1)

        # ----------------------------
        # CRITIC UPDATE (Quantile Huber)
        # ----------------------------
        with torch.no_grad():
            next_action = self.actor(next_state)
            
            # Target Quantiles: [B, M, N]
            target_q = self.target_critic(next_state, next_action)
            
            # TQC Logic: Sort and Truncate (Remove overestimation from top)
            # Pool
            target_pooled = target_q.view(self.batch_size, -1)
            sorted_target, _ = torch.sort(target_pooled, dim=1)
            
            # Keep bottom 90% (Drop top 10% outliers)
            n_keep = int(sorted_target.shape[1] * 0.9)
            truncated_target = sorted_target[:, :n_keep]
            
            # Randomly sample back to N quantiles to keep shape consistent? 
            # OR just use the truncated set as the target distribution Z.
            # Usually TQC averages the truncated set to get scalar target, 
            # BUT for Distributional, we simply match the distribution.
            # To simplify implementation: We calculate Quantile Loss against the truncated set atoms.
            target_Z = rew + (1 - done) * self.gamma * truncated_target

        # Current Quantiles: [B, M, N]
        current_q = self.critic(state, act)
        
        # Calculate Quantile Huber Loss
        # We compute loss for each ensemble member against the target Z
        critic_loss = 0.0
        
        # Z shape: [B, K] where K is truncated count
        # Pred shape: [B, N] per member
        
        # We need pairwise differences: [B, N, 1] - [B, 1, K] = [B, N, K]
        target_Z_expanded = target_Z.unsqueeze(1) 
        
        for i in range(self.critic.ensemble_size):
            pred = current_q[:, i, :].unsqueeze(2) 
            diff = target_Z_expanded - pred
            
            # Huber
            huber = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
            
            # Quantile Weight (tau)
            # tau_i = (i + 0.5) / N
            tau = (torch.arange(self.n_quantiles).float().to(device) + 0.5) / self.n_quantiles
            tau = tau.view(1, -1, 1)
            
            # Pinball Loss
            element_loss = (tau - (diff.detach() < 0).float()).abs() * huber
            
            # Average over quantiles and batch, WEIGHTED by PER
            # Note: per_weights is [B, 1]
            # element_loss is [B, N, K]
            weighted_loss = element_loss.mean(dim=(1, 2)) * per_weights.squeeze()
            critic_loss += weighted_loss.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update PER Priorities
        # Use mean absolute TD error (approx) for priority
        # TD = E[Target] - E[Pred]
        with torch.no_grad():
            mean_target = target_Z.mean(dim=1)
            # Average over ensemble
            mean_pred = current_q.mean(dim=(1,2))
            td_errors = (mean_target - mean_pred).abs().cpu().numpy()
            
        self.memory.update_priorities(idxs, td_errors)

        # ----------------------------
        # ACTOR UPDATE (Maximize CVaR)
        # ----------------------------
        new_action = self.actor(state)
        
        # Get CVaR of pooling distribution
        cvar = self.get_ensemble_cvar(state, new_action, self.alpha_cvar)
        
        # Maximize CVaR
        actor_loss = -cvar.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft Update Targets
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
            
        return critic_loss.item(), actor_loss.item(), cvar.mean().item()

# -----------------------------------------------------------
# 4. MAIN TRAINING LOOP
# -----------------------------------------------------------

if __name__ == "__main__":
    # Create Config
    voi_configs18 = [
        ((6, 6, 6), 0.5, (3, 3, 3)),
        ((12, 12, 12), 0.8, (3, 3, 3)),
        ((9, 4, 9), 0.6, (3, 3, 3)),
        ((9, 15, 9), 0.4, (3, 3, 3)),
        ((9, 9, 3), 0.7, (3, 3, 3)),
        ((9, 9, 15), 0.9, (3, 3, 3)),
    ]
    base_config = {
        "volume_shape": (18, 18, 18),
        "target_center": (9, 9, 9),
        "target_size": (3, 3, 3),
        "vois": voi_configs18,
    }

    config_env =  {
        "volume_shape": (18, 18, 18),
        "target_size": (3, 3, 3),
        "base_config": base_config,
        "source_distance": 9,
        "voi_configs": voi_configs18,
        "epsilon": 1e-3,
        "dose_target": 1.0,
        "max_beams": 3,
        "num_layers": 6,
        "raster_grid": (4, 4),
        "raster_spacing": (1.0, 1.0),
        "max_steps": 3,
    }

    # Initialize
    env = BeamAngleEnv(config_env)
    agent = RACER_TQC_PER(env, config_env)
    
    # Curriculum Settings
    TOTAL_EPISODES = 5000
    WARMUP_EPISODES = 2000
    
    scores = []
    
    print("--- STARTING RACER-TQC-PER TRAINING ---")

    for ep in range(TOTAL_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        # 1. Update Intensity Limit (Curriculum)
        # Linear ramp: 0.1 -> 1.0
        prog = min(ep / WARMUP_EPISODES, 1.0)
        #agent.actor.current_limit = 0.1 + (0.9 * prog)
        agent.actor.current_limit = 1.0
        frames = []
        while not done:
            action = agent.get_action(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(env.render())
            agent.memory.push(obs, action, reward, next_obs, done)
            
            stats = agent.update(per_beta=0.4 + 0.6*prog) # Anneal Beta for PER
            
            obs = next_obs
            episode_reward += reward
        
        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:])
        if ep %100 == 0:
            import imageio as iio
            iio.mimsave('eval.gif', frames, duration=1.0)
            print("Saved 'eval.gif'")
        if ep % 10 == 0:
            cvar_str = f"{stats[2]:.2f}" if stats else "None"
            print(f"Ep {ep} | Reward: {episode_reward:.2f} | Avg100: {avg_score:.2f} | Limit: {agent.actor.current_limit:.2f} | CVaR: {cvar_str}")
            
        # Checkpoint
        if ep % 500 == 0:
            torch.save(agent.actor.state_dict(), f"actor_ep{ep}.pth")

    print("Training Complete.")
    torch.save(agent.actor.state_dict(), "final_actor.pth")