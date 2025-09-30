# file: envs/pyRadPlanAngleEnv.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict 
import torch.nn as nn
import torch

# pyRadPlan imports (top-level API from README)
from pyRadPlan import load_patient, IonPlan, generate_stf, calc_dose_influence
from importlib import resources
import SimpleITK as sitk
from scipy.io import loadmat
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose
from scipy.linalg import null_space
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from customConvModel import CustomConv3DModel, CustomConv3DModelTF
import json
from io import BytesIO
import matplotlib.pyplot as plt
import imageio as iio
from pathlib import Path
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Custom4DConvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # observation_space.shape = (C, D, H, W)
        super().__init__(observation_space, features_dim)
        C, D, H, W = observation_space.shape

        self.conv = nn.Sequential(
            nn.Conv3d(C, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute conv output size
        with torch.no_grad():
            sample_input = torch.zeros(1, C, D, H, W)
            n_flatten = self.conv(sample_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.fc(self.conv(observations))


class PyRadPlanAngleEnv(gym.Env):
    """
    Gymnasium environment for selecting beam angles (Phase 1).
    - Discrete action over candidate beams (indices)
    - Observation: CT grid (downsampled) + structure masks + current cumulative dose (flattened)
    - Reward: coverage - OAR penalties (customizable)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 config):
        super().__init__()

        # --- Load patient data (example from README) ---
        patient_mat_resource = config["patient_mat_resource"]
        tg119_file = config["tg119_file"]
        candidate_gantry_angles = config["candidate_gantry_angles"]
        candidate_couch_angles = config["candidate_couch_angles"]
        self.max_beams = config["max_beams"]
        self.downsample_factor = config["downsample_factor"]
        tg119_path = resources.files(patient_mat_resource).joinpath(tg119_file)
        self.ct, self.cst = load_patient(tg119_path)
        self.cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
        self.cst.vois[0].objectives = [
        SquaredOverdosing(priority=10.0, d_max=1.0),
        SquaredOverdosing(priority=5.0, d_max=1.0, quantity="let_dose"),
        ]  # OAR
        self.cst.vois[2].objectives = [
            MeanDose(priority=1.0, d_ref=0.0),
            SquaredOverdosing(priority=10.0, d_max=2.0),
        ]  # BODY


        # Plan object (use to configure generators / engines)
        self.pln = IonPlan(radiation_mode="protons", machine="Generic")

        # candidate geometry: if None create default sets
        if candidate_gantry_angles is None:
            candidate_gantry_angles = list(np.linspace(0, 180, 12, endpoint=False))
        if candidate_couch_angles is None:
            #candidate_couch_angles = list(np.linspace(0, 180, 6, endpoint=False))  # expand if desired
            candidate_couch_angles = [0.0]  # common choices
        # Build list of candidate beams (cartesian product)
        self.beam_list = []
        for g in candidate_gantry_angles:
            for c in candidate_couch_angles:
                self.beam_list.append({"gantry": float(g), "couch": float(c)})
        self.num_candidates = len(self.beam_list)
        
        # --- Precompute / generate STF + dose-influence matrices for each candidate beam ---
        # (Important for speed: do once)
        # Use pyRadPlan's stf generator for IMPT to generate beam geometries for each candidate.
        self.stf_cache = []
        self.dij_cache = []  # each item: flattened dose influence vector for that beam

        # configure plan properties 
        self.pln.prop_stf = {"generator": "IMPT"}
    

        for i, b in enumerate(self.beam_list):
            
            # set angles in plan prop and call generate_stf(ct, cst, pln)
            self.pln.prop_stf.update({"gantry_angles": [b["gantry"]], "couch_angles": [b["couch"]]})
            stf = generate_stf(self.ct, self.cst, self.pln)  # stf describes the steering geometry for the beam(s)
            dij = calc_dose_influence(self.ct, self.cst, stf, self.pln)
            try:
                # make sure dij reports a positive number of beamlets
                n_bixels = getattr(dij, "total_num_of_bixels", None)
                print(f"[beam {i}] gantry={b['gantry']}, couch={b['couch']}, total_num_of_bixels={n_bixels}")

                if not n_bixels or n_bixels <= 0:
                    print(f"  -> WARNING: no beamlets for beam {i}; skipping this candidate.")
                    self.stf_cache.append(stf)
                    self.dij_cache.append(None)
                    continue
                # Attempt to compute a unit fluence dose to catch indexing errors early
                fluence = np.ones(int(n_bixels))
                beam_dose = dij.compute_result_ct_grid(fluence)  # may raise ValueError
                #  key/shape checks:
                if "physical_dose" not in beam_dose:
                    print(f"  -> WARNING: compute_result_ct_grid returned no 'physical_dose' for beam {i}")
                    self.stf_cache.append(stf)
                    self.dij_cache.append(None)
                    continue

            except Exception as e:
                print(f"  -> EXCEPTION while precomputing beam {i}: {e}")
                # store placeholder to indicate invalid beam
                self.stf_cache.append(stf)
                self.dij_cache.append(None)
                continue

            # If everything passed, store dij
            self.stf_cache.append(stf)
            self.dij_cache.append(dij)



            
            
        # Observation space: here we provide a simple numeric vector:
        # - downsampled CT values + flattened current dose (both flattened) + binary structure masks summary
        # NOTE: adapt this to the state you want (e.g., full 3D arrays -> huge). We recommend low-dim first.
        self.ct_array = sitk.GetArrayFromImage(self.ct.cube_hu)  # (z,y,x)
        ct_shape = self.ct_array.shape  # (X, Y, Z)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2, *ct_shape),  # 2 channels: [CT, dose]
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.num_candidates)

        # Runtime state
        self.current_dose = np.zeros(ct_shape, dtype=np.float32)
        self.selected_beams = []
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_dose[:] = 0.0
        self.selected_beams = []
        self.steps_taken = 0
        return self._get_obs(), {}

    def _get_obs(self):
        ct_vol = self.ct_array.astype(np.float32)
        dose_vol = self.current_dose.astype(np.float32)
        obs = np.stack([ct_vol, dose_vol], axis=0)  # (2, X, Y, Z)
        return obs

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        if action in self.selected_beams:
            reward = -0.1
            done = False
            info = {"note": "beam_already_selected", "selected_beams": self.selected_beams}
            return self._get_obs(), reward, done, False, info

        #self.selected_beams.append(action)
        #self.steps_taken += 1

        dij = self.dij_cache[action]
        if dij is None:
        # Penalize invalid beam and don't add dose
            reward = -1.0
            self.steps_taken += 1
            self.selected_beams.append(action)
            done = self.steps_taken >= self.max_beams
            info = {"note": "invalid_beam", "action": action, "selected_beams": self.selected_beams}
            return self._get_obs(), float(reward), done, False, info    
        
        n_bixels = int(getattr(dij, "total_num_of_bixels", 0))
        if n_bixels <= 0:
            reward = -1.0
            self.selected_beams.append(action)
            self.steps_taken += 1
            done = self.steps_taken >= self.max_beams
            info = {"note": "zero_beamlets", "action": action}
            return self._get_obs(), float(reward), done, False, info    
        
        fluence = np.ones(dij.total_num_of_bixels)  # correct number of beamlets for this Dij
        # second dimenion = #beamlets
        try:
            beam_dose = dij.compute_result_ct_grid(fluence)
        except Exception as e:
            print(f"⚠️ compute_result_ct_grid ValueError for action {action}: {e}")
            reward = -1.0
            self.selected_beams.append(action)
            self.steps_taken += 1
            done = self.steps_taken >= self.max_beams
            info = {"note": "compute_failed", "error": str(e)}
            return self._get_obs(), float(reward), done, False, info
        
        physical_dose_array = sitk.GetArrayFromImage(beam_dose["physical_dose"])
        self.current_dose += np.array(physical_dose_array, dtype=np.float32)

        reward = self._compute_reward()
        self.selected_beams.append(action)
        self.steps_taken += 1
        done = self.steps_taken >= self.max_beams
        info = {"selected_beams": self.selected_beams}
        truncated = False
        return self._get_obs(), float(reward), done, False, info

    def _compute_reward(self):
        # Placeholder reward: mean target dose - mean OAR dose
        try:
            target_mask = self.cst.get_structure_mask("PTV")
        except AttributeError:
            target_mask = np.zeros_like(self.current_dose, dtype=bool)

        oar_masks = []
        for name in ["OAR1", "OAR2"]:
            try:
                m = self.cst.get_structure_mask(name)
                oar_masks.append(m)
            except Exception:
                pass

        mean_target = self.current_dose[target_mask].mean() if target_mask.any() else 0.0
        mean_oars = np.mean([self.current_dose[m].mean() for m in oar_masks]) if oar_masks else 0.0
        reward = mean_target - 0.8 * mean_oars
        if np.isnan(reward):
          return 0.0
        else:
          return  reward
    
    def get_mean_dose(self, voi_name: str) -> float:
        for voi in self.cst.vois:
            if voi.name.lower() == voi_name.lower():
                mask = sitk.GetArrayFromImage(voi.mask).astype(bool)
                return float(self.current_dose[mask].mean())
        return 0.0

    def compute_reward(self):
        """
        Compute reward based on objectives attached to cst.vois.
        Higher reward = better plan.
        """
        total_penalty = 0.0

        # loop over all structures (VOIs)
        for voi in self.cst.vois:
            if not voi.objectives:
                continue  # skip if no objectives defined

            # mask of voxels belonging to this VOI
            mask = sitk.GetArrayFromImage(voi.mask).astype(bool)

            # dose values in those voxels
            dose_vals = self.current_dose[mask]

            # evaluate each objective
            for obj in voi.objectives:
                try:
                    penalty = obj(dose_vals)  # pyRadPlan objectives are callable
                except Exception as e:
                    print(f"⚠️ Warning: could not evaluate {obj} on {voi.name}: {e}")
                    penalty = 0.0

                total_penalty += penalty

        # reward = negative of penalty
        reward = -float(total_penalty)

        return reward


    def render(self, mode="human"):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    config_env = {"patient_mat_resource": "pyRadPlan.data.phantoms",
                 "tg119_file":  "TG119.mat",
                 "candidate_gantry_angles": None,
                 "candidate_couch_angles":None,
                 "max_beams": 5,
                 "downsample_factor":  4}
    
    env = PyRadPlanAngleEnv(config_env)
    obs, info = env.reset()
    print("Obs shape:", obs.shape, "Action space:", env.action_space)

    # Train  PPO just to test
    # Policy kwargs
    policy_kwargs = dict(
    features_extractor_class=Custom4DConvExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    )

# Initialize PPO with custom policy
    model = PPO(
    "CnnPolicy",       # base policy wrapper
    env,               # your environment
    policy_kwargs=policy_kwargs,
    verbose=1,
    device='cuda'      # GPU usage
    )
    model.learn(total_timesteps=100000)

    # Evaluate
    obs, info = env.reset()
    done = False
    step = 0
    while not done and step< 10:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print("Step reward:", reward)
        done = terminated or truncated
        step+=1
    
    
