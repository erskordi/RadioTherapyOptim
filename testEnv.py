import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import json
import random
import torch
import tensorflow as tf
from io import BytesIO
from pprint import pprint

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
from customConvModel import CustomConv3DModel, CustomConv3DModelTF

class CustomEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.depth, self.height, self.width = self.config['depth'], self.config['height'], self.config['width']
        self.max_colors = self.config['max_colors']

        self.target_state = np.zeros_like(
            np.reshape(np.arange(self.depth * self.height * self.width), 
                       [self.depth, self.height, self.width]), dtype=np.float32)
        self.target_state[self.depth//2,1:self.height,1:self.width] = 1.0

        self.observation_space = spaces.Box(
            0, 1, 
            shape=(self.depth, self.height, self.width), dtype=np.float32
        )

        # Define the action space as a multidiscrete space
        self.action_space = spaces.Box(
            0, 1, 
            shape=(self.depth * self.height * self.width,), dtype=np.float32
        )

        self.max_step = self.config['horizon']

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.state = self.observation_space.sample().round()
        
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        action = action.reshape(self.depth, self.height, self.width).round()
        # Ensure action is within bounds
        action = np.clip(action, 0, 1)

        # Update the state based on the action
        self.state = action
        
        # Calculate the reward as the Euclidean distance between the current state and the target state
        #reward = - np.sum(np.abs(self.state.flatten() - self.target_state.flatten()) ** 2)
        #reward = - np.linalg.norm(self.state - self.target_state) #** 2
        reward = - (1 - np.sum(self.state.flatten() == self.target_state.flatten()) / self.state.size)

        terminated = truncated = True if np.isclose(reward, 0.0, rtol=1e-1, atol=0) or self.step_count >= self.max_step else False # else True
        
        info = {
            'step_count': self.step_count,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated
        }

        return self.state, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        # Visualize the state as a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.depth)
        ax.set_ylim(0, self.height)
        ax.set_zlim(0, self.width)
        x, y, z = np.where(self.state > 0.5)
        ax.scatter(x, y, z)
        # add the target state
        x_t, y_t, z_t = np.where(self.target_state > 0.5)
        ax.scatter(x_t, y_t, z_t, c='red', marker='o')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        return (img * 255).astype(np.uint8)

    def close(self):
        plt.close()

def test_run(env_name):
    env = env_name
    obs, _ = env.reset()
    frames = []
    terminated = False

    while not terminated:
        action = env.action_space.sample().round()
        print("Action:", action)
        obs, reward, terminated, _, _ = env.step(action)
        print("Observation:", obs)
        print("Target state:", env.target_state)
        print("Reward:", reward)
        print("Done:", terminated)
        
        frames.append(env.render())
        

    iio.mimsave('voxel_environment.gif', frames, duration=0.3)

    env.close()

if __name__ == "__main__":
    test = False
    evaluation = False
    training = not test and not evaluation

    env_config = {
            'depth': 3,
            'height': 3,
            'width': 3,
            'max_colors': 255,
            'horizon': 100,
        }
    
    
    # Test the environment
    if test:
        test_run(CustomEnv(env_config))
    elif training:
        ray.init()

        # Register the custom model
        ModelCatalog.register_custom_model(
            "CustomConv3DModel",
            CustomConv3DModel,
        )

        ModelCatalog.register_custom_model(
            "conv3d_model_tf",
              CustomConv3DModelTF
        )

        # Register the custom environment
        register_env("CustomEnv-v0", lambda config: CustomEnv(env_config))
        # Define RLlib configuration for PPO
        config = (
            PPOConfig()
            .framework("tf2")
            .environment("CustomEnv-v0")
            .env_runners(
                num_env_runners=6, 
                preprocessor_pref=None, # The default gave nan observations after some point
                #num_envs_per_env_runner=10,
                #batch_mode="complete_episodes",
                #rollout_fragment_length=80,
            )
            .training(
                model={
                    "custom_model": "conv3d_model_tf",
                    "vf_share_layers": True,
                },
                #entropy_coeff=0.001,
                entropy_coeff_schedule=[
                    [0, 0.2],
                    [1000, 0.1],
                    [5000, 0.01],
                    [8000, 0.001],
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
        '''
        # Build the algorithm.
        algo = config.build_algo()
        
        # Train it for 5 iterations ...
        for _ in range(5):
            pprint(algo.train())
        
        # ... and evaluate it.
        pprint(algo.evaluate())

        # Release the algo's resources (remote actors, like EnvRunners and Learners).
        algo.stop()
        '''
        tuner = tune.Tuner(
            "PPO",
            run_config=train.RunConfig(
                stop={"training_iteration": 2000},
                checkpoint_config=train.CheckpointConfig(
                        checkpoint_at_end=True,
                        checkpoint_frequency=10,),
                verbose=2,
            ),
            param_space=config,
        )
        results = tuner.fit()
    
        # Get the best trial
        best_trial = results.get_best_result("episode_reward_mean", mode="max", scope="all")
        if best_trial is not None:
            print(f'Best trial: {best_trial.checkpoint}')
        else:
            print('No best trial found.')
        
        path_to_policy = best_trial.checkpoint.path + '/policies/default_policy'
        
        ray.shutdown()

        ray.init(_system_config={
                "local_fs_capacity_threshold": 0.99,
                "object_spilling_config": json.dumps(
                    {
                    "type": "filesystem",
                    "params": {
                        "directory_path": "/tmp/spill",
                        "buffer_size": 1_000_000,
                    }
                    },
                )
            },)
        
        # Register the custom environment
        register_env("CustomEnv-v0", lambda config: CustomEnv())

        ModelCatalog.register_custom_model(
            "CustomConv3DModel",
            CustomConv3DModel,
        )

        ModelCatalog.register_custom_model(
            "conv3d_model_tf",
              CustomConv3DModelTF
        )

        #path_to_policy = '/Users/erotokritosskordilis/ray_results/PPO_2025-04-27_11-51-15/PPO_CustomEnv-v0_736df_00000_0_2025-04-27_11-51-15/checkpoint_000282' + '/policies/default_policy'
        # Evaluate policy
        
        rl_module = Policy.from_checkpoint(
            path_to_policy
            )
        env = CustomEnv(env_config)
        # Environment sanity test
        obs, _ = env.reset()
        frames = []
        frames.append(env.render())
        terminated = truncated = False
        
        # Test step
        for _ in range(5):
            action = np.clip(rl_module.compute_single_action(obs, explore=False)[0], 0, 1)
            #action = np.clip(action, 0, 100)
            obs, reward, terminated, truncated, info = env.step(action)

            frames.append(env.render())
        ''''''
        # Save the frames as a GIF
        iio.mimsave('eval.gif', frames, duration=1)

        env.close()
        ray.shutdown()