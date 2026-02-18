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
from clinical_env import BeamAngleEnv

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

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

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        # We store everything as numpy arrays or floats to save memory
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack
        states, actions, rewards, next_states, dones = zip(*batch)
        #returns a tuple of batch_size elements
        
        # Helper to stack the dictionary observations
        def stack_obs(obs_list):
            return torch.FloatTensor(np.stack(obs_list)).to(self.device)

        return (
            stack_obs(states),
            torch.FloatTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device),
            stack_obs(next_states),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------
# 2. NETWORKS
# -----------------------------------------------------------

class ResidualBlock3D(nn.Module):
    """
    A single residual block for 3D data.
    Structure: Input -> ReLU -> Conv3d -> ReLU -> Conv3d -> Add Input
    """
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        
        
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        
        return out + residual  # The Skip Connection

class ImpalaBlock3D(nn.Module):
    """
    One stage of the IMPALA architecture.
    Structure: Conv3d (channel adjust) -> MaxPool -> ResBlock -> ResBlock
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Two residual blocks per stage (standard IMPALA configuration)
        self.res1 = ResidualBlock3D(out_channels)
        self.res2 = ResidualBlock3D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class DeepResNetEncoder(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.flat_size = 128 * 2 * 2 * 2 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)


class EnsembleQuantileCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, n_quantiles=51, ensemble_size=5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_quantiles = n_quantiles
        #one shared encoder
        self.encoder = DeepResNetEncoder()
        input_dim = self.encoder.flat_size +  action_dim
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n_quantiles)) for _ in range(ensemble_size)
        ])

    def forward(self, state, action):
        img_feat = self.encoder(state)
        x = torch.cat([img_feat, action], dim=1)
        return torch.stack([net(x) for net in self.nets], dim=1)

class Actor(nn.Module):
    def __init__(self, obs_shape, action_dim, action_low, action_high):
        super().__init__()
        self.encoder = DeepResNetEncoder()
        input_dim = self.encoder.flat_size
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.register_buffer("low", torch.tensor(action_low))
        self.register_buffer("high", torch.tensor(action_high))

    def forward(self, state):
        x = self.encoder(state)
        return torch.tanh(self.net(x))

# -----------------------------------------------------------
# 3. RACER AGENT
# -----------------------------------------------------------

class RACER_Quantile_PER:
    def __init__(self, env, config):
        self.last_actor_loss = 0.0  # Storage for logging
        self.last_cvar = 0.0
        self.policy_delay = 2# Storage for logging
        self.step_counter = 0
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha_cvar = 0.1
        self.n_quantiles = 25
        self.batch_size = 256
        self.lr = 3e-4
        self.noise_start = 0.2
        self.noise_end = 0.1  # Exploration floor
        self.noise_decay = 0.999 # Decay rate
        self.current_noise = self.noise_start
        # Store bounds as tensors for the update loop
        self.action_low_tensor = torch.FloatTensor(env.action_space.low).to(device)
        self.action_high_tensor = torch.FloatTensor(env.action_space.high).to(device)
        
        # Store bounds as numpy for get_action (CPU)
        self.action_low_numpy = env.action_space.low
        self.action_high_numpy = env.action_space.high
        
        self.obs_shape = env.observation_space.shape
        self.action_dim = env.action_space.shape[0]
        
        self.actor = Actor(self.obs_shape, self.action_dim, env.action_space.low, env.action_space.high).to(device)
        self.target_actor = Actor(self.obs_shape, self.action_dim, env.action_space.low, env.action_space.high).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict()) # Copy weights
        self.critic = EnsembleQuantileCritic(self.obs_shape, self.action_dim, self.n_quantiles).to(device)
        self.target_critic = EnsembleQuantileCritic(self.obs_shape, self.action_dim, self.n_quantiles).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.memory = ReplayBuffer(500000, device = device)

    def get_action(self, state, explore=True):
        self.actor.eval()
        dose = torch.FloatTensor(state).to(device)
        if dose.dim() == 4:  # [C, D, H, W] -> [1, C, D, H, W]
            dose = dose.unsqueeze(0)
        with torch.no_grad():
            # 1. Get Normalized Action [-1, 1]
            raw_action = self.actor(dose).cpu().numpy()[0]
        
        if explore:
            # 2. Add Noise to [-1, 1] space
            noise = np.random.normal(0, self.current_noise, size=raw_action.shape)
            raw_action = np.clip(raw_action + noise, -1.0, 1.0)
            
        
        # 3. Scale to Environment Units
        # Formula: low + 0.5 * (raw + 1) * (high - low)
        # This is mathematically identical to your manual scaling for angles,
        # and treats energy/intensity (0-1) correctly if their low=0, high=1.
        scaled_action = self.action_low_numpy + 0.5 * (raw_action + 1.0) * (self.action_high_numpy - self.action_low_numpy)
        
        return scaled_action

    def get_ensemble_cvar(self, state, action, alpha):
        quantiles = self.critic(state, action)  # [B, Ensemble, N_Q]
    
        cvars = []
        for i in range(self.critic.ensemble_size):
            q_i = quantiles[:, i, :]  # [B, N_Q]
            sorted_q, _ = torch.sort(q_i, dim=1)
            k = max(1, int(sorted_q.shape[1] * alpha))
            cvar_i = sorted_q[:, :k].mean(dim=1, keepdim=True)  # [B, 1]
            cvars.append(cvar_i)
    
        #   Average CVaR across ensemble
        stacked = torch.stack(cvars, dim=1)  # [B, Ensemble, 1]
        return stacked.mean(dim=1)  # [B, 1]

    def update(self):
        self.step_counter += 1
        if len(self.memory) < self.batch_size:
            return None
            
        state, action, reward, next_state, done= self.memory.sample(self.batch_size)
        
        # Critic Update
        with torch.no_grad():
            #policy smoothing
            next_raw_action = self.target_actor(next_state) # Returns [-1, 1]
            noise = (torch.randn_like(next_raw_action) * 0.2).clamp(-0.5, 0.5)
            # Scale next_action before passing to target_critic
            next_smooth = (next_raw_action + noise).clamp(-1.0, 1.0)
            next_action = self.action_low_tensor + 0.5 * (next_smooth + 1.0) * (self.action_high_tensor - self.action_low_tensor)
            # B. Truncated Target (Distributional "Min")
            target_q = self.target_critic(next_state, next_action) # [Batch, Ensemble, N_Q]
            target_pooled = target_q.view(self.batch_size, -1)         # [Batch, Total_Atoms]
            sorted_z, _ = torch.sort(target_pooled, dim=1)
        
        # Keep bottom 80% (removes optimistic outliers)
            d=0.8
            n_keep = int(sorted_z.shape[1] * d) 
            target_trunc = sorted_z[:, :n_keep]
        
            target_Z = reward + (1 - done) * self.gamma * target_trunc # [Batch, n_keep]
        # C. Calculate Loss
        current_q = self.critic(state, action) # [Batch, Ensemble, N_Q]
        
        # Reshape for broadcasting
        # Target: [Batch, 1, n_keep]
        target_Z_exp = target_Z.unsqueeze(1)
        
        critic_loss = 0.0 
        
        for i in range(self.critic.ensemble_size):
            pred = current_q[:, i, :].unsqueeze(2) #[Batch, N_Q, 1]
            diff = target_Z_exp - pred# [Batch, N_Q, n_keep]
            huber = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
            tau = (torch.arange(self.n_quantiles).float().to(device) + 0.5) / self.n_quantiles
            tau = tau.view(1, -1, 1)
            element_loss = (tau - (diff.detach() < 0).float()).abs() * huber
            #weighted_loss = element_loss.mean(dim=(1, 2)) * per_weights.squeeze()
            critic_loss += element_loss.sum(dim=1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        

        # delayed Actor Update
        if self.step_counter % self.policy_delay == 0:
            new_action = self.actor(state)
            new_action = self.action_low_tensor + 0.5 * (new_action + 1.0) * (self.action_high_tensor - self.action_low_tensor)
            cvar = self.get_ensemble_cvar(state, new_action, self.alpha_cvar)
            actor_loss = -cvar.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
        
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
            for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
            self.last_cvar = cvar.mean().item()
            self.last_actor_loss = actor_loss.item()
        return critic_loss.item(), self.last_actor_loss, self.last_cvar

# -----------------------------------------------------------
# 4. MAIN TRAINING LOOP
# -----------------------------------------------------------
SAVE_DIR = "clinical"
os.makedirs(SAVE_DIR, exist_ok=True)
if __name__ == "__main__":
    voi_configs18 = [
        ((6, 6, 6), 0.5, (3, 3, 3)), ((12, 12, 12), 0.8, (3, 3, 3)), ((9, 4, 9), 0.6, (3, 3, 3)),
        ((9, 15, 9), 0.4, (3, 3, 3)), ((9, 9, 3), 0.7, (3, 3, 3)), ((9, 9, 15), 0.9, (3, 3, 3)),
    ]
    base_config = {"volume_shape": (18, 18, 18), "target_center": (9, 9, 9), "target_size": (3, 3, 3), "vois": voi_configs18}
    config_env =  {"volume_shape": (18, 18, 18), "target_size": (3, 3, 3), "base_config": base_config, 
                   "source_distance": 9, "voi_configs": voi_configs18, "epsilon": 1e-3, "dose_target": 1.0, 
                   "max_beams": 10, "num_layers": 6, "raster_grid": (4, 4), "raster_spacing": (1.0, 1.0), "max_steps": 10}

    env = BeamAngleEnv(config_env)
    agent = RACER_Quantile_PER(env, config_env)
    
    TOTAL_EPISODES = 10000
    
    scores = []
    cvar_history = []
    start_steps = 1000
    
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
            if ep < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs, explore=True)
                
        
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append(env.render())
            
            agent.memory.push(obs, action, reward, next_obs, done)
            
            if ep>= start_steps:
                stats = agent.update()
            
                if stats is not None:
                    ep_cvar_accum += stats[2]
                    updates_count += 1
            else:
                stats = None
            obs = next_obs
            episode_reward += reward
        if ep>= start_steps:
            agent.current_noise = max(agent.noise_end, agent.current_noise * agent.noise_decay)    
        # End of Episode Stats
        scores.append(episode_reward)
        avg_cvar = (ep_cvar_accum / updates_count) if updates_count > 0 else None
        cvar_history.append(avg_cvar)
        
        avg_score = np.mean(scores[-100:])
        
        if ep % 100 == 0:
            # [FIX] Run a dedicated EVALUATION episode (No Noise, No Training)
            eval_obs, _ = env.reset() # Use standard or fixed config for fair comparison
            eval_done = False
            eval_frames = []
            
            while not eval_done:
                # [FIX] explore=False ensures we see what the agent actually learned
                eval_action = agent.get_action(eval_obs, explore=False) 
                eval_next_obs, _, eval_term, eval_trunc, _ = env.step(eval_action)
                eval_done = eval_term or eval_trunc
                eval_frames.append(env.render())
                eval_obs = eval_next_obs
            
            # Save the CLEAN gif
            iio.mimsave(f'{SAVE_DIR}/eval_ep{ep}.gif', eval_frames, duration=3.0)
            #save_plots(scores, cvar_history, filename=f'{SAVE_DIR}/learning_curves_ep{ep}.png')
            
            # Log
            log_str = f"Ep {ep} | Reward: {episode_reward:.2f} | Avg100: {avg_score:.2f} | Noise: {agent.current_noise:.3f}"
            if stats is not None:
                critic_loss, actor_loss, _ = stats
                log_str += f" | CVaR: {avg_cvar:.3f} | ActLoss: {actor_loss:.3f} | CritLoss: {critic_loss:.3f}"
            print(log_str)
            
        if ep % 500 == 0:
            torch.save(agent.actor.state_dict(), f"actor_ep{ep}.pth", file_name=f"{SAVE_DIR}/actor_ep{ep}.pth")
        
        if ep ==5000:
            for pg in agent.actor_optimizer.param_groups:
                pg['lr'] = 1e-4
            for pg in agent.critic_optimizer.param_groups:
                pg['lr'] = 1e-4

    print("Training Complete.")
    torch.save(agent.actor.state_dict(), "final_actor.pth", file_name=f"{SAVE_DIR}/final_actor.pth")
    save_plots(scores, cvar_history, filename=f"{SAVE_DIR}/final_learning_curves.png")