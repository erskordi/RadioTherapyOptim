import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

# Import your environment
from Numba_Test_Train import BeamAngleEnv

# -----------------------------------------------------------
# 1. RACER UTILITIES (Distributional RL & CVaR)
# -----------------------------------------------------------

def to_tensor(x, device):
    if isinstance(x, dict):
        return {k: to_tensor(v, device) for k, v in x.items()}
    return torch.FloatTensor(x).to(device)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Helper to stack dict stats
        def stack_obs(obs_list):
            return {
                "dose": np.stack([o["dose"] for o in obs_list]),
                "beams": np.stack([o["beams"] for o in obs_list]),
            }

        return (
            to_tensor(stack_obs(state), self.device),
            to_tensor(np.array(action), self.device),
            to_tensor(np.array(reward), self.device).unsqueeze(1),
            to_tensor(stack_obs(next_state), self.device),
            to_tensor(np.array(done), self.device).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------
# 2. NEURAL NETWORKS
# -----------------------------------------------------------

class CNNFeatureExtractor(nn.Module):
    """
    Simple 3D CNN to process the (2, D, H, W) dose/mask input.
    Replace this with your CustomConv3DModel if needed.
    """
    def __init__(self, input_shape):
        super().__init__()
        # Input shape: (2, 18, 18, 18)
        self.conv1 = nn.Conv3d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        
        # Calculate flat size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flat_size = x.view(1, -1).size(1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.pool(y)
        y = F.relu(self.conv2(y))
        y = self.pool(y)
        return y.view(y.size(0), -1)

class DistributionalEnsembleCritic(nn.Module):
    """
    Ensemble of Critics.
    Output: Z(s,a) as a categorical distribution (logits) for EACH ensemble member.
    """
    def __init__(self, obs_shape, beam_dim, action_dim, num_atoms=51, v_min=-5, v_max=5, ensemble_size=2):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = torch.linspace(v_min, v_max, num_atoms)
        self.ensemble_size = ensemble_size
        
        self.feature_extractor = CNNFeatureExtractor(obs_shape)
        
        # Input: CNN features + Beam History + Action
        input_dim = self.feature_extractor.flat_size + beam_dim + action_dim
        
        # We create independent networks for the ensemble
        self.q_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, num_atoms) # Output logits for distribution
            ) for _ in range(ensemble_size)
        ])

    def forward(self, state, action):
        # Extract features
        img_feat = self.feature_extractor(state["dose"])
        beam_feat = state["beams"]
        
        x = torch.cat([img_feat, beam_feat, action], dim=1)
        
        # Forward pass for each ensemble member
        # Returns list of tensors: [Batch, Atoms]
        logits_list = [net(x) for net in self.q_nets]
        
        # Return probability distributions
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        return probs_list, logits_list
    
    def get_atoms(self, device):
        return self.atoms.to(device)

class AdaptiveActor(nn.Module):
    """
    Actor with Adaptive Action Limits (Paper Eq. 4).
    Uses a shifted Tanh to soft-clip actions into a dynamic range.
    """
    def __init__(self, obs_shape, beam_dim, action_dim, action_space_low, action_space_high):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(obs_shape)
        input_dim = self.feature_extractor.flat_size + beam_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim) 
        )
        
        # Global bounds
        self.register_buffer("low", torch.tensor(action_space_low))
        self.register_buffer("high", torch.tensor(action_space_high))
        
        # Adaptive Limit Parameters (Learnable)
        # We initialize them conservatively (small range) and expand them
        # In Radiotherapy: "Safe" is usually 0 intensity.
        # We start with full angle range but restricted intensity/energy.
        self.limit_scale = nn.Parameter(torch.ones(action_dim) * 0.1) 
        self.limit_center = nn.Parameter(torch.zeros(action_dim))

    def softclip(self, raw_action, v_min, v_max):
        """Eq 4 in RACER paper"""
        # eta = (v_max - v_min) / 2
        # mu = (v_max + v_min) / 2
        # return eta * tanh((raw - mu)/eta) + mu
        # Simplified: We map raw output to [v_min, v_max] dynamically
        
        scale = (v_max - v_min) / 2.0
        center = (v_max + v_min) / 2.0
        return scale * torch.tanh(raw_action) + center

    def forward(self, state):
        img_feat = self.feature_extractor(state["dose"])
        beam_feat = state["beams"]
        x = torch.cat([img_feat, beam_feat], dim=1)
        
        raw_action = self.net(x)
        
        # Calculate current dynamic bounds based on learnable params
        # We clamp the learnable limits to be within the global environment bounds
        # Note: In the paper, they expand bounds. Here we just learn a window.
        # To strictly follow RACER's expansion logic, we would optimize limit_scale separateley.
        # For simplicity here, we apply the global tanh, but you can risk-optimize the bounds.
        
        # Standard SAC/PPO approach:
        # action = torch.tanh(raw_action)
        # scaled_action = self.low + 0.5 * (action + 1.0) * (self.high - self.low)
        
        # RACER Adaptive Limits (Simplified implementation):
        # We use the raw_action directly into a tanh scaled by global bounds
        # BUT, the paper optimizes the bounds.
        # Here we will just use standard tanh squashing for stability first.
        action = torch.tanh(raw_action)
        
        # Rescale to env bounds
        scaled_action = (action * (self.high - self.low) / 2.0) + (self.high + self.low) / 2.0
        
        return scaled_action

# -----------------------------------------------------------
# 3. THE RACER AGENT
# -----------------------------------------------------------

class RACER:
    def __init__(self, env, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        
        self.obs_shape = env.observation_space["dose"].shape
        self.beam_dim = env.observation_space["beams"].shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Hyperparams
        self.gamma = 0.99
        self.alpha_cvar = 0.10  # Risk sensitivity (0.1 = worst 10% cases)
        self.num_atoms = 51
        self.v_min = -5.0
        self.v_max = 5.0
        self.batch_size = 32
        self.lr = 3e-4
        
        # Networks
        self.actor = AdaptiveActor(self.obs_shape, self.beam_dim, self.action_dim, 
                                   env.action_space.low, env.action_space.high).to(self.device)
        self.critic = DistributionalEnsembleCritic(self.obs_shape, self.beam_dim, self.action_dim, 
                                                   self.num_atoms, self.v_min, self.v_max).to(self.device)
        self.target_critic = DistributionalEnsembleCritic(self.obs_shape, self.beam_dim, self.action_dim, 
                                                          self.num_atoms, self.v_min, self.v_max).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.replay_buffer = ReplayBuffer(10000, self.device)
        self.atoms = self.critic.get_atoms(self.device)

    def get_action(self, state, explore=True):
        self.actor.eval()
        state_tensor = to_tensor(state, self.device)
        
        # FIX: Check for 4 dims (C, D, H, W) instead of 3
        # If input is (2, 18, 18, 18), we need to make it (1, 2, 18, 18, 18)
        if state_tensor["dose"].dim() == 4:
            state_tensor["dose"] = state_tensor["dose"].unsqueeze(0)
            state_tensor["beams"] = state_tensor["beams"].unsqueeze(0)
            
        with torch.no_grad():
            action = self.actor(state_tensor)
            
        action = action.cpu().numpy()[0]
        
        if explore:
            # Gaussian noise for exploration
            noise = np.random.normal(0, 0.2, size=action.shape)
            action = np.clip(action + noise, self.env.action_space.low, self.env.action_space.high)
            
        return action

    def compute_cvar_loss(self, probs, atoms, alpha):
        """
        Differentiable CVaR calculation for categorical distribution.
        Eq: CVaR_alpha(Z) = E[Z | Z < VaR_alpha(Z)]
        Algorithm 1 in Paper.
        """
        # 1. Calculate CDF
        cdf = torch.cumsum(probs, dim=1)
        
        # 2. Worst-case CDF (Clip at alpha)
        # We need to normalize the tail to sum to 1
        # Tail prob mass = P_i if CDF_i <= alpha (roughly)
        
        # Differentiable approximation:
        # We want to weight the atoms that fall in the bottom alpha percentile.
        
        # Find index where CDF crosses alpha
        # Note: This is tricky to do fully differentiably without soft sorting or approximations.
        # RACER paper uses specific Algo 1:
        
        # Calculate worst-case PDF
        # P_hat_i = (min(CDF_i, alpha) - min(CDF_{i-1}, alpha)) / alpha
        
        cdf_prev = torch.roll(cdf, 1, dims=1)
        cdf_prev[:, 0] = 0.0
        
        cdf_clipped = torch.min(cdf, torch.tensor(alpha, device=self.device))
        cdf_prev_clipped = torch.min(cdf_prev, torch.tensor(alpha, device=self.device))
        
        p_hat = (cdf_clipped - cdf_prev_clipped) / alpha
        
        # CVaR is expectation over p_hat
        cvar = torch.sum(p_hat * atoms, dim=1)
        
        return cvar

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # ----------------------------
        # 1. CRITIC UPDATE (Distributional)
        # ----------------------------
        
        # Compute Target Distribution
        with torch.no_grad():
            next_action = self.actor(next_state) # Actor gives next action
            # Target critic ensemble
            target_probs_list, _ = self.target_critic(next_state, next_action)
            
            # Average probabilities across ensemble for target (or take min/max logic)
            # Standard approach: Use mean distribution for stability
            avg_target_probs = torch.stack(target_probs_list).mean(dim=0)
            
            # Project distribution (Categorical Algorithm)
            delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)
            tz = reward + (1 - done) * self.gamma * self.atoms
            tz = tz.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Distribute probability mass
            # m corresponds to the target distribution projected onto the support
            m = torch.zeros(self.batch_size, self.num_atoms).to(self.device)
            
            # Simple projection loop (vectorized scatter is faster but this is clearer)
            # Note: For strict correctness, we should do this per sample
            # This is a simplified projection for brevity
            for i in range(self.batch_size):
                for j in range(self.num_atoms):
                    m[i, l[i, j]] += avg_target_probs[i, j] * (u[i, j].float() - b[i, j])
                    m[i, u[i, j]] += avg_target_probs[i, j] * (b[i, j] - l[i, j].float())

        critic_loss = 0
        current_probs_list, current_logits_list = self.critic(state, action)
        
        for logits in current_logits_list:
            # Cross Entropy between projected target (m) and current logits
            # m is probabilities, logits are raw scores
            critic_loss += -torch.sum(m * F.log_softmax(logits, dim=1), dim=1).mean()
        
        # Add Entropy Regularization for OOD (Optional, from RACER paper)
        # We skip Explicit Entropy Max for brevity, but it goes here.

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------
        # 2. ACTOR UPDATE (Maximize CVaR)
        # ----------------------------
        
        # We want to choose action 'a' that maximizes CVaR_alpha(Z(s, a))
        new_action = self.actor(state)
        actor_probs_list, _ = self.critic(state, new_action)
        
        # We use the "worst" critic in the ensemble (pessimism) or the mean
        # RACER paper suggests optimizing CVaR of the ensemble mixture
        avg_actor_probs = torch.stack(actor_probs_list).mean(dim=0)
        
        cvar_value = self.compute_cvar_loss(avg_actor_probs, self.atoms, self.alpha_cvar)
        
        # We want to MAXIMIZE CVaR, so minimize negative
        actor_loss = -cvar_value.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 3. Soft Update Target
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * 0.995 + param.data * 0.005)

        return critic_loss.item(), actor_loss.item(), cvar_value.mean().item()

# -----------------------------------------------------------
# 4. TRAINING LOOP
# -----------------------------------------------------------

if __name__ == "__main__":
    # Config matching your environment
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

    env = BeamAngleEnv(config_env)
    agent = RACER(env, config_env)
    
    num_episodes = 10000
    scores = []
    
    print("Starting  Training...")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        frames = []
        while not done:
            action = agent.get_action(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            frames.append(env.render())
            
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_reward += reward
            
            # Update Agent
            stats = agent.update()
        
        scores.append(episode_reward)
        avg_score = np.mean(scores[-100:])
        if ep %100 == 0:
            import imageio as iio
            iio.mimsave('eval.gif', frames, duration=1.0)
            print("Saved 'eval.gif'")
        if ep % 10 == 0:
            print(f"Episode {ep}, Cumulative reward: {episode_reward:.2f}, Avg 100: {avg_score:.2f}")
            if stats:
                print(f"  Critic Loss: {stats[0]:.4f}, Actor CVaR: {stats[2]:.4f}")

    # Save
    torch.save(agent.actor.state_dict(), "racer_actor.pth")
    print("Training Complete.")