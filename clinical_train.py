import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import imageio as iio

# Import Clinical Environment
from clinical_env import ClinicalRadiotherapyEnv

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Create run directory
RUN_DIR = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(f"{RUN_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{RUN_DIR}/gifs", exist_ok=True)
os.makedirs(f"{RUN_DIR}/plots", exist_ok=True)
print(f"Saving results to: {RUN_DIR}")


# -----------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------

def save_plots(scores, cvar_history, filename=None):
    if filename is None:
        filename = f"{RUN_DIR}/plots/learning_curves.png"
    
    window = 50
    
    def moving_average(data, win):
        if len(data) < win:
            return data
        return np.convolve(data, np.ones(win), 'valid') / win

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(scores, alpha=0.3, color='blue', label='Raw Reward')
    if len(scores) >= window:
        ma_scores = moving_average(scores, window)
        ax1.plot(np.arange(len(ma_scores)) + window - 1, ma_scores, color='blue', linewidth=2, label=f'Avg ({window})')
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    clean_cvar = [c if c is not None else -10.0 for c in cvar_history]
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


def save_training_metrics(coverage_history, violation_history, eval_interval, filename=None):
    if filename is None:
        filename = f"{RUN_DIR}/plots/clinical_metrics.png"
    
    if len(coverage_history) < 2:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    x = np.arange(len(coverage_history)) * eval_interval
    
    ax1.plot(x, coverage_history, 'g-', linewidth=2)
    ax1.axhline(y=95, color='r', linestyle='--', label='Target (95%)')
    ax1.set_ylabel("Coverage (%)")
    ax1.set_title("Target Coverage During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    ax2.plot(x, violation_history, 'r-', linewidth=2)
    ax2.axhline(y=0, color='g', linestyle='--', label='Goal (0)')
    ax2.set_ylabel("OAR Violations")
    ax2.set_xlabel("Episode")
    ax2.set_title("OAR Violations During Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# -----------------------------------------------------------
# REPLAY BUFFER
# -----------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        def stack_obs(obs_list):
            return {
                "dose": torch.FloatTensor(np.stack([o["dose"] for o in obs_list])).to(self.device),
                "step": torch.FloatTensor(np.stack([o["step"] for o in obs_list])).to(self.device)
            }

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
# NETWORKS
# -----------------------------------------------------------

class ResidualBlock3D(nn.Module):
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
        return out + residual


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
    def __init__(self, obs_shape, step_dim, action_dim, n_quantiles=25, ensemble_size=5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.n_quantiles = n_quantiles
        self.encoder = DeepResNetEncoder()
        input_dim = self.encoder.flat_size + step_dim + action_dim
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, n_quantiles)
            ) for _ in range(ensemble_size)
        ])

    def forward(self, state, action):
        img_feat = self.encoder(state["dose"])
        step_feat = state["step"]
        x = torch.cat([img_feat, step_feat, action], dim=1)
        return torch.stack([net(x) for net in self.nets], dim=1)


class Actor(nn.Module):
    def __init__(self, obs_shape, step_dim, action_dim, action_low, action_high):
        super().__init__()
        self.encoder = DeepResNetEncoder()
        input_dim = self.encoder.flat_size + step_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.register_buffer("low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, state):
        img_feat = self.encoder(state["dose"])
        step_feat = state["step"]
        x = torch.cat([img_feat, step_feat], dim=1)
        return torch.tanh(self.net(x))


# -----------------------------------------------------------
# RACER AGENT
# -----------------------------------------------------------

class RACER_Agent:
    def __init__(self, env, config):
        self.last_actor_loss = 0.0
        self.last_cvar = 0.0
        self.policy_delay = 2
        self.step_counter = 0
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha_cvar = 0.1
        self.n_quantiles = 25
        self.batch_size = 256
        self.lr = 3e-4
        self.noise_start = 0.2
        self.noise_end = 0.05
        self.noise_decay = 0.9995
        self.current_noise = self.noise_start
        
        self.action_low_tensor = torch.FloatTensor(env.action_space.low).to(device)
        self.action_high_tensor = torch.FloatTensor(env.action_space.high).to(device)
        self.action_low_numpy = env.action_space.low
        self.action_high_numpy = env.action_space.high
        
        self.obs_shape = env.observation_space["dose"].shape
        self.step_dim = env.observation_space["step"].shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.actor = Actor(
            self.obs_shape, self.step_dim, self.action_dim,
            env.action_space.low, env.action_space.high
        ).to(device)
        
        self.target_actor = Actor(
            self.obs_shape, self.step_dim, self.action_dim,
            env.action_space.low, env.action_space.high
        ).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = EnsembleQuantileCritic(
            self.obs_shape, self.step_dim, self.action_dim, self.n_quantiles
        ).to(device)
        
        self.target_critic = EnsembleQuantileCritic(
            self.obs_shape, self.step_dim, self.action_dim, self.n_quantiles
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.memory = ReplayBuffer(500000, device=device)

    def get_action(self, state, explore=True):
        self.actor.eval()
        dose = torch.FloatTensor(state["dose"]).to(device)
        step = torch.FloatTensor(state["step"]).to(device)
        
        if dose.dim() == 4:
            dose = dose.unsqueeze(0)
        if step.dim() == 1:
            step = step.unsqueeze(0)
        
        state_tensor = {"dose": dose, "step": step}
        
        with torch.no_grad():
            raw_action = self.actor(state_tensor).cpu().numpy()[0]
        
        if explore:
            noise = np.random.normal(0, self.current_noise, size=raw_action.shape)
            raw_action = np.clip(raw_action + noise, -1.0, 1.0)
        
        scaled_action = self.action_low_numpy + 0.5 * (raw_action + 1.0) * (self.action_high_numpy - self.action_low_numpy)
        return scaled_action

    def get_ensemble_cvar(self, state, action, alpha):
        quantiles = self.critic(state, action)
        cvars = []
        for i in range(self.critic.ensemble_size):
            q_i = quantiles[:, i, :]
            sorted_q, _ = torch.sort(q_i, dim=1)
            k = max(1, int(sorted_q.shape[1] * alpha))
            cvar_i = sorted_q[:, :k].mean(dim=1, keepdim=True)
            cvars.append(cvar_i)
        stacked = torch.stack(cvars, dim=1)
        return stacked.mean(dim=1)

    def update(self):
        self.step_counter += 1
        if len(self.memory) < self.batch_size:
            return None
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Critic Update
        with torch.no_grad():
            next_raw_action = self.target_actor(next_state)
            noise = (torch.randn_like(next_raw_action) * 0.2).clamp(-0.5, 0.5)
            next_smooth = (next_raw_action + noise).clamp(-1.0, 1.0)
            next_action = self.action_low_tensor + 0.5 * (next_smooth + 1.0) * (self.action_high_tensor - self.action_low_tensor)
            
            target_q = self.target_critic(next_state, next_action)
            target_pooled = target_q.view(self.batch_size, -1)
            sorted_z, _ = torch.sort(target_pooled, dim=1)
            
            d = 0.8
            n_keep = int(sorted_z.shape[1] * d)
            target_trunc = sorted_z[:, :n_keep]
            
            target_Z = reward + (1 - done) * self.gamma * target_trunc
        
        current_q = self.critic(state, action)
        
        critic_loss = 0.0
        for i in range(self.critic.ensemble_size):
            pred = current_q[:, i, :].unsqueeze(2)
            target_Z_exp = target_Z.unsqueeze(1)
            diff = target_Z_exp - pred
            huber = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
            tau = (torch.arange(self.n_quantiles).float().to(device) + 0.5) / self.n_quantiles
            tau = tau.view(1, -1, 1)
            element_loss = (tau - (diff.detach() < 0).float()).abs() * huber
            critic_loss += element_loss.sum(dim=1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Delayed Actor Update
        if self.step_counter % self.policy_delay == 0:
            new_raw_action = self.actor(state)
            new_action = self.action_low_tensor + 0.5 * (new_raw_action + 1.0) * (self.action_high_tensor - self.action_low_tensor)
            cvar = self.get_ensemble_cvar(state, new_action, self.alpha_cvar)
            actor_loss = -cvar.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            # Soft update targets
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
            for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
            
            self.last_cvar = cvar.mean().item()
            self.last_actor_loss = actor_loss.item()
        
        return critic_loss.item(), self.last_actor_loss, self.last_cvar


# -----------------------------------------------------------
# CHECKPOINTING
# -----------------------------------------------------------

def save_checkpoint(agent, episode, final=False):
    name = "final" if final else f"ep{episode}"
    path = f"{RUN_DIR}/checkpoints/checkpoint_{name}.pth"
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "target_actor": agent.target_actor.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "noise": agent.current_noise,
        "step_counter": agent.step_counter
    }, path)
    print(f">>> Saved checkpoint: {path}")


def load_checkpoint(agent, path):
    checkpoint = torch.load(path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.target_actor.load_state_dict(checkpoint["target_actor"])
    agent.target_critic.load_state_dict(checkpoint["target_critic"])
    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    agent.current_noise = checkpoint.get("noise", agent.noise_end)
    agent.step_counter = checkpoint.get("step_counter", 0)
    print(f">>> Loaded checkpoint: {path}")


# -----------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------

def evaluate(agent, env, n_episodes=5, save_gif=False, gif_name=None, verbose=False):
    results = {
        "rewards": [],
        "coverages": [],
        "violations": [],
        "critical_violations": [],
        "acceptable": [],
        "steps": []
    }
    
    all_frames = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        frames = []
        
        while not done:
            action = agent.get_action(obs, explore=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if save_gif and ep == 0:
                frames.append(env.render())
            
            obs = next_obs
            episode_reward += reward
        
        eval_result = env.evaluate()
        summary = eval_result["summary"]
        
        results["rewards"].append(episode_reward)
        results["coverages"].append(summary["target_coverage"])
        results["violations"].append(summary["n_oar_violations"])
        results["critical_violations"].append(summary["critical_violations"])
        results["acceptable"].append(summary["plan_acceptable"])
        results["steps"].append(env.step_count)
        
        if save_gif and ep == 0:
            all_frames = frames
        
        if verbose:
            status = "✓" if summary["plan_acceptable"] else "✗"
            print(f"  Eval {ep+1}: Reward={episode_reward:.2f}, Coverage={summary['target_coverage']:.1f}%, "
                  f"Violations={summary['n_oar_violations']}, Steps={env.step_count} [{status}]")
    
    if save_gif and len(all_frames) > 0:
        if gif_name is None:
            gif_name = f"{RUN_DIR}/gifs/evaluation.gif"
        iio.mimsave(gif_name, all_frames, duration=0.5)
    
    return {
        "avg_reward": np.mean(results["rewards"]),
        "avg_coverage": np.mean(results["coverages"]),
        "avg_violations": np.mean(results["violations"]),
        "avg_critical": np.mean(results["critical_violations"]),
        "acceptable_rate": np.mean(results["acceptable"]),
        "avg_steps": np.mean(results["steps"]),
        "raw": results
    }


def evaluate_all_scenarios(agent, config_base, n_episodes=10, verbose=True):
    scenarios = ["head_neck", "prostate", "lung"]
    all_results = {}
    
    print("\n" + "="*60)
    print("MULTI-SCENARIO EVALUATION")
    print("="*60)
    
    for scenario in scenarios:
        config = config_base.copy()
        config["scenario"] = scenario
        env = ClinicalRadiotherapyEnv(config)
        
        print(f"\n--- {scenario.upper()} ---")
        results = evaluate(
            agent, env, n_episodes=n_episodes, verbose=verbose,
            save_gif=True, gif_name=f"{RUN_DIR}/gifs/eval_{scenario}.gif"
        )
        all_results[scenario] = results
        
        print(f"  Avg Reward:     {results['avg_reward']:.2f}")
        print(f"  Avg Coverage:   {results['avg_coverage']:.1f}%")
        print(f"  Avg Violations: {results['avg_violations']:.1f}")
        print(f"  Acceptable:     {results['acceptable_rate']*100:.0f}%")
        print(f"  Avg Steps:      {results['avg_steps']:.1f}")
    
    print("\n" + "="*60)
    print(f"{'Scenario':<12} {'Coverage':>10} {'Violations':>12} {'Acceptable':>12}")
    print("-"*60)
    for scenario, res in all_results.items():
        print(f"{scenario:<12} {res['avg_coverage']:>9.1f}% {res['avg_violations']:>12.1f} {res['acceptable_rate']*100:>11.0f}%")
    print("="*60)
    
    return all_results


# -----------------------------------------------------------
# TRAINING
# -----------------------------------------------------------

def train(agent, env, total_episodes=10000, start_steps=1000, eval_interval=100, save_interval=500):
    scores = []
    cvar_history = []
    coverage_history = []
    violation_history = []
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Scenario: {env.scenario_name}")
    print(f"Device: {device}")
    print(f"Action dim: {agent.action_dim}")
    print(f"Total episodes: {total_episodes}")
    print(f"Warmup episodes: {start_steps}")
    print("="*60 + "\n")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        ep_cvar_accum = 0.0
        updates_count = 0
        
        while not done:
            if ep < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs, explore=True)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(obs, action, reward, next_obs, done)
            
            if ep >= start_steps:
                stats = agent.update()
                if stats is not None:
                    ep_cvar_accum += stats[2]
                    updates_count += 1
            
            obs = next_obs
            episode_reward += reward
        
        # Decay exploration
        if ep >= start_steps:
            agent.current_noise = max(agent.noise_end, agent.current_noise * agent.noise_decay)
        
        scores.append(episode_reward)
        avg_cvar = (ep_cvar_accum / updates_count) if updates_count > 0 else None
        cvar_history.append(avg_cvar)
        
        # Periodic evaluation and logging
        if ep % eval_interval == 0:
            save_gif = (ep % (eval_interval * 5) == 0)
            eval_results = evaluate(
                agent, env, n_episodes=3,
                save_gif=save_gif,
                gif_name=f"{RUN_DIR}/gifs/eval_ep{ep}.gif"
            )
            coverage_history.append(eval_results["avg_coverage"])
            violation_history.append(eval_results["avg_violations"])
            
            save_plots(scores, cvar_history)
            save_training_metrics(coverage_history, violation_history, eval_interval)
            
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            
            print(f"Ep {ep:5d} | Reward: {episode_reward:7.2f} | Avg100: {avg_score:7.2f} | "
                  f"Noise: {agent.current_noise:.3f} | Coverage: {eval_results['avg_coverage']:5.1f}% | "
                  f"Violations: {eval_results['avg_violations']:.1f} | Acceptable: {eval_results['acceptable_rate']*100:.0f}%")
        
        # Save checkpoints
        if ep % save_interval == 0 and ep > 0:
            save_checkpoint(agent, ep)
        
        # Learning rate decay at midpoint
        if ep == total_episodes // 2:
            for pg in agent.actor_optimizer.param_groups:
                pg['lr'] = 1e-4
            for pg in agent.critic_optimizer.param_groups:
                pg['lr'] = 1e-4
            print(">>> Learning rate reduced to 1e-4")
    
    print("\n--- TRAINING COMPLETE ---")
    save_checkpoint(agent, total_episodes, final=True)
    save_plots(scores, cvar_history)
    
    return scores, cvar_history, coverage_history, violation_history


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    # Environment configuration
    config_env = {
        "volume_shape": (18, 18, 18),
        "raster_grid": (4, 4),
        "raster_spacing": (1.0, 1.0),
        "max_steps": 5,
        "num_layers": 6,
        "scenario": "prostate",
        "randomize_scenario": False,
        "apply_uncertainty": True,
        "range_uncertainty_std": 0.03,
        "setup_uncertainty_std": 0.3,
        "coverage_weight": 5.0,
        "constraint_weight": 2.0,
        "critical_violation_penalty": 5.0,
    }
    
    # Create environment and agent
    env = ClinicalRadiotherapyEnv(config_env)
    agent = RACER_Agent(env, config_env)
    
    # Train
    scores, cvar_history, coverage_history, violation_history = train(
        agent, env,
        total_episodes=10000,
        start_steps=1000,
        eval_interval=100,
        save_interval=500
    )
    
    # Final evaluation on all scenarios
    print("\n\nFinal Evaluation on All Scenarios:")
    evaluate_all_scenarios(agent, config_env, n_episodes=20, verbose=False)
    
    # Print detailed evaluation for training scenario
    print("\n\nDetailed Evaluation (Training Scenario):")
    env.reset()
    for _ in range(env.max_steps):
        action = agent.get_action(env.reset()[0], explore=False)
        env.step(action)
    env.print_evaluation()