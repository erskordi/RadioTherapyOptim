import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from collections import deque
import matplotlib.pyplot as plt
import imageio as iio

# Import  Numba Environment
from Numba_Test_Train import BeamAngleEnv

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# -----------------------------------------------------------
# 1. UTILS (Prioritized Experience Replay + Plotting)
# -----------------------------------------------------------

def save_plots(scores, cvar_history, filename="learning_curves.png"):
    """
    Plots Cumulative Return and CVaR curves with moving averages (to have a smoother curve).
    """
    window = 50
    
    # Calculate Moving Averages
    def moving_average(data, win):
        if len(data) < win: return data
        return np.convolve(data, np.ones(win), 'valid') / win

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # 1. Reward Plot
    ax1.plot(scores, alpha=0.3, color='blue', label='Raw Reward')
    if len(scores) >= window:
        ma_scores = moving_average(scores, window)
        ax1.plot(np.arange(len(ma_scores)) + window - 1, ma_scores, color='blue', linewidth=2, label=f'Avg ({window})')
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. CVaR Plot
    # Filter out None values (from warmup) for plotting
    clean_cvar = [c if c is not None else -10.0 for c in cvar_history] # Fill None with -10
    
    ax2.plot(clean_cvar, alpha=0.3, color='red', label='Raw CVaR')
    if len(clean_cvar) >= window:
        ma_cvar = moving_average(clean_cvar, window)
        ax2.plot(np.arange(len(ma_cvar)) + window - 1, ma_cvar, color='red', linewidth=2, label=f'Avg ({window})')
    ax2.set_ylabel("CVaR (Risk metric)")
    ax2.set_xlabel("Episode")
    ax2.set_title("Critic CVaR Estimate (Safety)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class SumTree:
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
        if self.write >= self.capacity: self.write = 0
        if self.n_entries < self.capacity: self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree): break
            if s <= self.tree[left]: idx = left
            else:
                s -= self.tree[left]
                idx = right
        idx = min(idx, len(self.tree) - 1)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0: max_p = 1.0
        self.tree.add(max_p, (state, action, reward, next_state, done))

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        def stack_obs(obs_list):
            return {"dose": np.stack([o["dose"] for o in obs_list]), "beams": np.stack([o["beams"] for o in obs_list])}

        return (stack_obs(states), np.array(actions), np.array(rewards), stack_obs(next_states), np.array(dones), np.array(idxs), np.array(is_weight, dtype=np.float32))

    def update_priorities(self, idxs, errors):
        for idx, err in zip(idxs, errors):
            p = (abs(err) + 1e-5) ** self.alpha
            self.tree.update(idx, p)

# -----------------------------------------------------------
# 2. NETWORKS
# -----------------------------------------------------------

class DeepResNetEncoder(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        # Input: (Batch, 2, 18, 18, 18)
        
        # Layer 1: Keep size, find edges
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1) 
        # Output: (16, 18, 18, 18)
        
        # Layer 2: Downsample
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1) 
        # Output: (32, 9, 9, 9)
        
        # Layer 3: Deep features
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1) 
        # Output: (64, 5, 5, 5)

        self.flat_size = 64 * 5 * 5 * 5  # = 8000 features

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)

class EnsembleQuantileCritic(nn.Module):
    def __init__(self, obs_shape, beam_dim, action_dim, n_quantiles=25, ensemble_size=2):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_quantiles = n_quantiles
        self.encoder = DeepResNetEncoder()
        input_dim = self.encoder.flat_size + beam_dim + action_dim
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n_quantiles)) for _ in range(ensemble_size)
        ])

    def forward(self, state, action):
        img_feat = self.encoder(state["dose"])
        beam_feat = state["beams"]
        x = torch.cat([img_feat, beam_feat, action], dim=1)
        return torch.stack([net(x) for net in self.nets], dim=1)

class Actor(nn.Module):
    def __init__(self, obs_shape, beam_dim, action_dim, action_low, action_high):
        super().__init__()
        self.encoder = DeepResNetEncoder()
        input_dim = self.encoder.flat_size + beam_dim
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.register_buffer("low", torch.tensor(action_low))
        self.register_buffer("high", torch.tensor(action_high))

    def forward(self, state):
        img_feat = self.encoder(state["dose"])
        beam_feat = state["beams"]
        x = torch.cat([img_feat, beam_feat], dim=1)
        
        raw_action = self.net(x)
        
        # Split logic
        geo_dim = 2; energy_dim = 6
        angles = torch.tanh(raw_action[:, :geo_dim])
        energies = torch.sigmoid(raw_action[:, geo_dim:geo_dim+energy_dim])
        intensities = torch.sigmoid(raw_action[:, geo_dim+energy_dim:])
        
        combined = torch.cat([angles, energies, intensities], dim=1)
        
        # Scale
        scaled = combined * 1.0 
        angle_low = self.low[:geo_dim]; angle_high = self.high[:geo_dim]
        scaled[:, :geo_dim] = (angles * (angle_high - angle_low) / 2.0) + (angle_high + angle_low) / 2.0
        return scaled

# -----------------------------------------------------------
# 3. RACER AGENT
# -----------------------------------------------------------

class RACER_Quantile_PER:
    def __init__(self, env, config):
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha_cvar = 0.10
        self.n_quantiles = 25
        self.batch_size = 64
        self.lr = 3e-4
        
        self.obs_shape = env.observation_space["dose"].shape
        self.beam_dim = env.observation_space["beams"].shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.actor = Actor(self.obs_shape, self.beam_dim, self.action_dim, env.action_space.low, env.action_space.high).to(device)
        self.critic = EnsembleQuantileCritic(self.obs_shape, self.beam_dim, self.action_dim, self.n_quantiles).to(device)
        self.target_critic = EnsembleQuantileCritic(self.obs_shape, self.beam_dim, self.action_dim, self.n_quantiles).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.memory = PrioritizedReplayBuffer(50000)

    def get_action(self, state, explore=True):
        self.actor.eval()
        dose = torch.FloatTensor(state["dose"]).to(device)
        beams = torch.FloatTensor(state["beams"]).to(device)
        if dose.dim() == 4: 
            dose = dose.unsqueeze(0)   
            beams = beams.unsqueeze(0) 
        state_tensor = {"dose": dose, "beams": beams}
        with torch.no_grad():
            action = self.actor(state_tensor)
        action = action.cpu().numpy()[0]
        if explore:
            noise = np.random.normal(0, 0.15, size=action.shape)
            action = np.clip(action + noise, env.action_space.low, env.action_space.high)
        return action

    def get_ensemble_cvar(self, state, action, alpha):
        quantiles = self.critic(state, action)
        pooled = quantiles.view(quantiles.shape[0], -1)
        sorted_q, _ = torch.sort(pooled, dim=1)
        k = max(1, int(sorted_q.shape[1] * alpha))
        tail = sorted_q[:, :k]
        return tail.mean(dim=1, keepdim=True)

    def update(self, per_beta=0.4):
        if self.memory.tree.n_entries < self.batch_size:
            return None
            
        states, actions, rewards, next_states, dones, idxs, weights = self.memory.sample(self.batch_size, per_beta)
        
        def to_tensor(x): return torch.FloatTensor(x).to(device)
        s_dose = to_tensor(states["dose"]); s_beams = to_tensor(states["beams"])
        state = {"dose": s_dose, "beams": s_beams}
        
        ns_dose = to_tensor(next_states["dose"]); ns_beams = to_tensor(next_states["beams"])
        next_state = {"dose": ns_dose, "beams": ns_beams}
        
        act = to_tensor(actions); rew = to_tensor(rewards).unsqueeze(1)
        done = to_tensor(dones).unsqueeze(1); per_weights = to_tensor(weights).unsqueeze(1)

        # Critic Update
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q = self.target_critic(next_state, next_action)
            target_pooled = target_q.view(self.batch_size, -1)
            target_Z = rew + (1 - done) * self.gamma * target_pooled

        current_q = self.critic(state, act)
        critic_loss = 0.0
        target_Z_expanded = target_Z.unsqueeze(1) 
        
        for i in range(self.critic.ensemble_size):
            pred = current_q[:, i, :].unsqueeze(2) 
            diff = target_Z_expanded - pred
            huber = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
            tau = (torch.arange(self.n_quantiles).float().to(device) + 0.5) / self.n_quantiles
            tau = tau.view(1, -1, 1)
            element_loss = (tau - (diff.detach() < 0).float()).abs() * huber
            weighted_loss = element_loss.mean(dim=(1, 2)) * per_weights.squeeze()
            critic_loss += weighted_loss.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update PER
        with torch.no_grad():
            mean_target = target_Z.mean(dim=1)
            mean_pred = current_q.mean(dim=(1,2))
            td_errors = (mean_target - mean_pred).abs().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)

        # Actor Update
        new_action = self.actor(state)
        cvar = self.get_ensemble_cvar(state, new_action, self.alpha_cvar)
        actor_loss = -cvar.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
            
        return critic_loss.item(), actor_loss.item(), cvar.mean().item()

# -----------------------------------------------------------
# 4. MAIN TRAINING LOOP
# -----------------------------------------------------------

if __name__ == "__main__":
    voi_configs18 = [
        ((6, 6, 6), 0.5, (3, 3, 3)), ((12, 12, 12), 0.8, (3, 3, 3)), ((9, 4, 9), 0.6, (3, 3, 3)),
        ((9, 15, 9), 0.4, (3, 3, 3)), ((9, 9, 3), 0.7, (3, 3, 3)), ((9, 9, 15), 0.9, (3, 3, 3)),
    ]
    base_config = {"volume_shape": (18, 18, 18), "target_center": (9, 9, 9), "target_size": (3, 3, 3), "vois": voi_configs18}
    config_env =  {"volume_shape": (18, 18, 18), "target_size": (3, 3, 3), "base_config": base_config, 
                   "source_distance": 9, "voi_configs": voi_configs18, "epsilon": 1e-3, "dose_target": 1.0, 
                   "max_beams": 3, "num_layers": 6, "raster_grid": (4, 4), "raster_spacing": (1.0, 1.0), "max_steps": 3}

    env = BeamAngleEnv(config_env)
    agent = RACER_Quantile_PER(env, config_env)
    
    TOTAL_EPISODES = 5000
    
    scores = []
    cvar_history = []
    
    print("--- STARTING RACER TRAINING (With Plots) ---")

    for ep in range(TOTAL_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        # Track avg CVaR per episode
        ep_cvar_accum = 0.0
        updates_count = 0
        
        frames = []
        
        while not done:
            action = agent.get_action(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(env.render())
            
            agent.memory.push(obs, action, reward, next_obs, done)
            
            beta = min(1.0, 0.4 + 0.6 * (ep / TOTAL_EPISODES))
            stats = agent.update(per_beta=beta)
            
            if stats is not None:
                ep_cvar_accum += stats[2]
                updates_count += 1
            
            obs = next_obs
            episode_reward += reward
        
        # End of Episode Stats
        scores.append(episode_reward)
        avg_cvar = (ep_cvar_accum / updates_count) if updates_count > 0 else None
        cvar_history.append(avg_cvar)
        
        avg_score = np.mean(scores[-100:])
        
        if ep % 100 == 0:
            iio.mimsave('eval.gif', frames, duration=1.0)
            save_plots(scores, cvar_history)
            print(f"Ep {ep} | Reward: {episode_reward:.2f} | Avg100: {avg_score:.2f} | CVaR: {avg_cvar}")
            
        # Checkpoint
        if ep % 500 == 0:
            torch.save(agent.actor.state_dict(), f"actor_ep{ep}.pth")

    print("Training Complete.")
    torch.save(agent.actor.state_dict(), "final_actor.pth")
    save_plots(scores, cvar_history)