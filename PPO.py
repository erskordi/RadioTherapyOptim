import random
from scipy.linalg import null_space
import numpy as np
import imageio
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import ray
from ray import tune, serve, train
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env, get_trainable_cls
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule
)
from Numba_Test_Train import BeamAngleEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# We assume this file exists, as per your original code
from customConvModelDict import CustomConv3DModel# CustomConvModel4DMTF
import json
from io import BytesIO
import matplotlib.pyplot as plt
import imageio as iio
import torch
from pathlib import Path
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
class GifCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        iteration = result["training_iteration"]

        if iteration % 200== 0:
            env = BeamAngleEnv(config_env)
            obs, _ = env.reset()
            frames = []
            frames.append(env.render())
            terminated = truncated = False

            while not (terminated or truncated):
                action = algorithm.compute_single_action(obs, explore=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                frames.append(env.render())
                
            

            script_dir = Path(__file__).resolve().parent
            out_dir = script_dir / "gifs"
            out_dir.mkdir(exist_ok=True)
            gif_path = out_dir / f"training_iter_{iteration}.gif"


            iio.mimsave(gif_path, frames, duration=2)
            print(f"Saved GIF at {gif_path}")

            env.close()
# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- Set test = True to verify the new sequential logic ---
    test = False
    evaluation = False
    training = not test and not evaluation

    
    voi_configs18 = [
        ((6, 6, 6), 0.5, (3, 3, 3)), ((12, 12, 12), 0.8, (3, 3, 3)), ((9, 4, 9), 0.6, (3, 3, 3)),
        ((9, 15, 9), 0.4, (3, 3, 3)), ((9, 9, 3), 0.7, (3, 3, 3)), ((9, 9, 15), 0.9, (3, 3, 3)),
    ]
    base_config = {"volume_shape": (18, 18, 18), "target_center": (9, 9, 9), "target_size": (3, 3, 3), "vois": voi_configs18}
    config_env =  {"volume_shape": (18, 18, 18), "target_size": (3, 3, 3), "base_config": base_config, 
                   "source_distance": 9, "voi_configs": voi_configs18, "epsilon": 1e-3, "dose_target": 1.0, 
                   "max_beams": 3, "num_layers": 6, "raster_grid": (4, 4), "raster_spacing": (1.0, 1.0), "max_steps": 3}

    
    
    # Test the sequential environment
    if test:
        print("--- RUNNING SEQUENTIAL TEST ---")
        env = BeamAngleEnv(config_env)
        obs, info = env.reset()
        print("Obs shape:", obs.shape)
        print("Action space (one beam):", env.action_space.shape)

        # --- Create 3 test actions ---
        actions = []
        for k in range(env.max_beams):
            action = env.action_space.sample() # Start with a random action
            # Make it deterministic for testing
            action[0] = k * (np.pi / 3) # gantry (0, 60, 120 deg)
            action[1] = 0.0 # couch
            # energies
            action[2: 2 + env.num_layers] = np.linspace(0.1, 0.9, env.num_layers)
            # intensities
            action[2 + env.num_layers:] = 0.5 # Set all intensities to 0.5
            actions.append(action)

        # --- Step sequentially ---
        frames = []
        frames.append(env.render()) # Initial state (Frame 0)
        
        total_reward = 0.0
        terminated = truncated = False
        
        while not terminated and not truncated:
            step_idx = env.step_count
            obs, reward, terminated, truncated, info = env.step(actions[step_idx])
            frames.append(env.render()) # Frame 1, 2, 3
            total_reward += reward
            print(f"Step {step_idx+1}/{env.max_steps}, Reward: {reward:.4f}")

        print(f"Total Episode Reward: {total_reward:.4f}")
        print("Final Dose max:", obs[0].max())
        
        iio.mimsave("sequential_test.gif", frames, duration=1.0)
        print("Saved 'sequential_test.gif'")
        print("--- TEST COMPLETE ---")

       
    elif training:
        print("--- STARTING TRAINING ---")
        ray.init()

        # Register the custom model
        ModelCatalog.register_custom_model(
             "CustomConv3DModel",
             CustomConv3DModel,
         )
        

        # Register the custom environment
        register_env("CustomEnv-v0", lambda config: BeamAngleEnv(config))
        
        # Define RLlib configuration for PPO
        config = (
            PPOConfig()
            .framework("torch")
            .environment("CustomEnv-v0", env_config=config_env)
            .callbacks(GifCallback)
            .resources(num_gpus=1) # Use 0 for CPU, 1 for GPU
            .env_runners(
                num_env_runners=9, # Number of parallel workers
                preprocessor_pref=None,
            )
            .training(
                model={
                    "custom_model": "CustomConv3DModel", # Uncomment if you have this
                    "vf_share_layers": True,
                },
                entropy_coeff_schedule=[
                    [0, 0.2],
                    [5000, 0.01],
                ],
                kl_coeff=0.3,
                kl_target=0.01,
                use_kl_loss=True,
                vf_clip_param=10.0,
                clip_param=0.2,
            )
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
        )
        
        tuner = tune.Tuner(
            "PPO",
            run_config=train.RunConfig(
                stop={"training_iteration":10000},
                checkpoint_config=train.CheckpointConfig(
                        checkpoint_at_end=True,
                        checkpoint_frequency=100,), # Save less often
                verbose=2,
            ),
            param_space=config,
        )
        results = tuner.fit()
    
        # --- EVALUATION ---
        best_trial = results.get_best_result("episode_reward_mean", mode="max", scope="all")
        if best_trial is not None:
            print(f'Best trial: {best_trial.checkpoint}')
            path_to_policy = best_trial.checkpoint.path + '/policies/default_policy'
        else:
            print('No best trial found.')
            ray.shutdown()
            exit()
        
        ray.shutdown()
        
        # --- Re-init for evaluation ---
        ray.init()
        
        register_env("CustomEnv-v0", lambda config: BeamAngleEnv(config))
        ModelCatalog.register_custom_model(
             "CustomConv3DModel",
             CustomConv3DModel,
         )
        
        rl_module = Policy.from_checkpoint(path_to_policy)
        env = BeamAngleEnv(config_env)
        
        # A new test config for evaluation
        test_config = {
            "target_center": (10, 10, 10), # Slightly different
            "target_size": (3, 3, 3),
            "vois": [
                ((7, 7, 7), 0.6, (3, 3, 3)),
                ((13, 13, 13), 0.7, (3, 3, 3)),
            ]
        }
    
        obs, _ = env.reset(config_override=test_config)
        frames = []
        terminated = False 
        truncated = False
        frames.append(env.render()) # Initial state
        
        final_reward = 0.0
        
        # --- UPDATED EVALUATION LOOP ---
        while not terminated and not truncated:
            # Use compute_single_action for Policy objects
            action = rl_module.compute_single_action(obs, explore=False)[0] 
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            final_reward = reward # Store the final reward of the last step

        print(f"Final evaluation reward: {final_reward:.4f}")
        iio.mimsave('eval.gif', frames, duration=1.0)
        print("Saved 'eval.gif'")

        env.close()
        ray.shutdown()