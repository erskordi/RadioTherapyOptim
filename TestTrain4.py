# beam_env_with_actor.py
"""
Sequential, multi-beam radiotherapy environment for reinforcement learning.

Features:
- Sequential beam placement: max_steps = max_beams.
- Agent adds ONE beam per step, observing the accumulated dose.
- Solvable action space: Agent controls one beam at a time.
- No 'active' parameter: Agent is forced to act, simplifying the problem.
- Dense reward: Rewards mean dose on target and penalizes inaction.
"""
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
import numba

# -----------------------
# Utility: ray-box intersection
# -----------------------
@numba.njit(fastmath=True)
def ray_box_intersection(p0, d, box_min, box_max, eps=1e-9):
    """
    function to compute the intersection between the beam and the volume of interest
    returns the time of intersection
    """
    tmin = np.zeros(3, dtype=float)
    tmax = np.zeros(3, dtype=float)
    for i in range(3):
        if abs(d[i]) < eps:
            # Ray is parallel to axis i
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return None # Ray is outside the slab and never intersects
            tmin[i] = -np.inf
            tmax[i] = np.inf
        else:
            t1 = (box_min[i] - p0[i]) / d[i]
            t2 = (box_max[i] - p0[i]) / d[i]
            tmin[i] = min(t1, t2)
            tmax[i] = max(t1, t2)
    t_entry = max(tmin)
    t_exit = min(tmax)
    if t_entry > t_exit or t_exit < 0:
        return None #No intersection
    t_entry_clipped = max(t_entry, 0.0)
    return t_entry_clipped, t_exit

# -----------------------
# Voxel traversal 
# -----------------------
# -----------------------
# Voxel traversal & Dose Loop (Numba-Jitted)
# -----------------------
@numba.njit(fastmath=True)
def _numba_voxel_loop(
    dose_array,       # The 3D dose grid to modify
    p0,               # Ray start point
    d,                # Ray direction
    grid_shape,       # e.g., (64, 64, 64) - THIS IS ALREADY A NP ARRAY
    t_entry,          # from ray_box_intersection
    t_exit,           # from ray_box_intersection
    current_intensity,
    peak_depth,
    volume_length      # for bragg_peak
):
    """
    This function combines traverse_grid_ray and the dose deposition
    into a single, fast Numba-jitted loop.
    It modifies dose_array in-place.
    """
    
    # --- Start of traverse_grid_ray logic ---
    p = p0 + t_entry * d
    voxel = np.floor(p).astype(np.int32)

    # Handle starting on max boundary
    for i in range(3):
        if p[i] == grid_shape[i] and d[i] < 0:
            voxel[i] = grid_shape[i] - 1
            
    # Clip to be safe (for rays starting inside)
    voxel_min = np.array([0, 0, 0], dtype=np.int32)
    
    # --- THIS IS THE FIX ---
    # grid_shape is already a numpy array, no need to re-wrap it
    voxel_max = grid_shape - 1
    
    for i in range(3):
        voxel[i] = max(voxel_min[i], min(voxel_max[i], voxel[i]))


    step = np.sign(d).astype(np.int32)
    t_delta = np.empty(3, dtype=np.float64)
    t_max = np.empty(3, dtype=np.float64)

    for i in range(3):
        if d[i] == 0:
            t_delta[i] = np.inf
            t_max[i] = np.inf
        else:
            if step[i] > 0:
                next_boundary = voxel[i] + 1.0
            else:
                next_boundary = float(voxel[i])
            t_max[i] = (next_boundary - p[i]) / d[i]
            t_delta[i] = abs(1.0 / d[i])

    t = t_entry
    
    # --- Main Voxel Loop ---
    while (0 <= voxel[0] < grid_shape[0] and
           0 <= voxel[1] < grid_shape[1] and
           0 <= voxel[2] < grid_shape[2] and
           t < t_exit + 1e-6):
        
        # --- Start of dose application logic ---
        x, y, z = voxel[0], voxel[1], voxel[2]
        voxel_center = np.array([x, y, z], dtype=np.float64) + 0.5
        
        depth_along_beam = np.dot(voxel_center - p0, d)
        
        if depth_along_beam >= t_entry - 1e-6:
            # Call our other jitted function
            dose_val = bragg_peak(
                depth=depth_along_beam,
                peak_depth=peak_depth,
                volume_lenght=volume_length
            )
            dose_array[x, y, z] += current_intensity * dose_val
        # --- End of dose application logic ---

        axis = np.argmin(t_max)
        t = t_max[axis]
        voxel[axis] += step[axis]
        t_max[axis] += t_delta[axis]

    # No return value needed, dose_array is modified in-place

# -----------------------
# Asymmetric Bragg-like peak (proton-like)
# -----------------------
@numba.njit(fastmath=True)  
def bragg_peak(depth, peak_depth, volume_lenght=18.0, width_pre=6.0, width_post=1.8, entrance_scale=0.2, tail_scale=0.05):
    """
    Toy depth-dose model along the beam ray:
    - small entrance dose before the peak,
    - Gaussian Bragg peak centered at `peak_depth`,
    - small distal tail after the peak.
    (Numba-optimized version for single float inputs)
    """
    width_pre = (volume_lenght * width_pre) / 250.0
    width_post = (volume_lenght * width_post) / 250.0

    if depth < 0.0:
        depth = 0.0

    # 1. Component for the rise of the peak (before the peak)
    rise_component = np.exp(-((max(0.0, peak_depth - depth))**2) / (2.0 * width_pre**2))

    # 2. Fall Component (after the peak)
    fall_component = np.exp(-((max(0.0, depth - peak_depth))**2) / (2.0 * width_post**2))

    # 3. tail component (starts at d=dp and decreases)
    tail = tail_scale * np.exp(-max(0.0, depth - peak_depth) / (width_post * 1.5))

    # A. Before the peak  (depth <= peak_depth): flat entrance  + increasing rise
    val_pre_peak = entrance_scale + (rise_component * (1.0 - entrance_scale))

    # B. After the peak (depth > peak_depth): Fall + tail
    val_post_peak = (fall_component * (1.0 - tail_scale)) + tail

    val = val_pre_peak if depth <= peak_depth else val_post_peak

    # Normalization factor is 1.0 (assuming peak is 1.0 by design)
    normfactor = 1.0

    return max(val * normfactor, 0.0)

@numba.njit(fastmath=True)
def orthonormal_plane_basis(v):
    # v: (3,)
    
    # v = np.asarray(v, dtype=float)  <--- THIS LINE IS REMOVED
    
    # Numba needs to know v is an array, but it should be already.
    # The rest of the function is Numba-compatible.
    if np.all(v == 0.0):
       raise ValueError("zero direction in orthonormal_plane_basis")
       
    if abs(v[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64) # Use np.float64
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64) # Use np.float64
        
    u = a - v * np.dot(a, v)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(v, u)
    w = w / (np.linalg.norm(w) + 1e-12)
    return u, w

# -----------------------
# Utility: Scene Perturbation
# -----------------------
def clamp_center(center, size, volume_shape):
    """
    Clamp a center so that a box of given `size` fully fits in `volume_shape`.
    """
    clamped = []
    for i in range(3):
        half = size[i] // 2
        # lowest possible center so box does not start < 0
        low = half
        # highest possible center so box does not extend beyond max index
        high = volume_shape[i] - (size[i] - half)
        # round, clamp
        c = int(round(center[i]))
        c = max(low, min(high, c))
        clamped.append(c)
    return tuple(clamped)

def perturb_config(base_config, volume_shape, max_shift=1, rng=None):
    """
    base_config: dict with keys "target_center","target_size","vois" 
    returns a new config with small integer shifts applied to centers, clipped inside volume.
    max_shift: maximum absolute integer shift in voxels per axis
    """
    if rng is None:
        rng = np.random.default_rng() # Use default generator if none provided

    # perturb target center by integer offsets
    tcenter = np.array(base_config["target_center"], dtype=int)
    
    
    shift = rng.integers(-max_shift, max_shift+1, size=3)
    
    
    size = base_config["target_size"]
    new_tcenter = clamp_center(tcenter + shift, size, volume_shape)

    # VOIs: perturb each center a bit and also jitter weight slightly (optional)
    new_vois = []
    for (c, w, size) in base_config["vois"]:
        c = np.array(c, dtype=int)
        
        # --- START FIX ---
        # Use .integers() instead of .randint()
        shift = rng.integers(-max_shift, max_shift+1, size=3)
        # --- END FIX ---
        
        new_c = clamp_center(c + shift,size, volume_shape)
        # jitter weight a little but keep it positive
        w_new = float(np.clip(w + rng.normal(scale=0.05), 0.05, 10.0))
        new_vois.append((new_c, w_new, size))

    return {
        "target_center": tuple(new_tcenter),
        "target_size": tuple(base_config["target_size"]),
        "vois": new_vois,
    }

class Beam:
    """
    Parallel-beam beamlet group:
    - raster grid of beamlet origins on a plane orthogonal to beam direction
    - all beamlets share same direction (parallel)
    -direction specified by gantry and couch angles (theta, phi)
    - apply_dose() method to compute dose deposition in volume for given intensities and energy
    """
    def __init__(self, gantry_angle, couch_angle, raster_grid_size, raster_spacing, volume_shape):
        self.gantry_angle = float(gantry_angle)
        self.couch_angle = float(couch_angle)
        self.raster_grid_size = tuple(map(int, raster_grid_size))
        self.raster_spacing = tuple(map(float, raster_spacing))
        self.volume_shape = tuple(map(int, volume_shape))
        self.volume_center = np.array(self.volume_shape, dtype=float) // 2.0 +0.5  # center of the volume in voxel coords
        self.source_distance = self.volume_shape[0]//2 +1 #distance from source to isocenter (approx volume center)
        self.direction = self._angles_to_direction(self.couch_angle, self.gantry_angle)  # unit vector
        
        self.source_point = self.get_source_point()
        
        self.raster_points = self.get_raster_points(center=self.source_point)

    def get_source_point(self):
        """
        Compute the source point along -direction, since will depend on the gantry angle in general the position of the source
        """
        source_point =  self.volume_center - self.source_distance * self.direction # we compute the source point given the direction and the center position
        return source_point
    
    def _angles_to_direction(self, theta, phi): 
        """ Convert spherical angles to a unit direction vector. """
        x = np.sin(theta) * np.cos(phi) 
        y = np.sin(theta) * np.sin(phi) 
        z = np.cos(theta) 
        v = np.array([x,y,z], dtype=float) 
        v /= np.linalg.norm(v) 
        return -v
    
    def get_raster_points(self, center=None):
        """
        Build a raster grid on the plane orthogonal to self.direction.
        The grid is centered on `center` (default: source_point).
        Raster points represent origins of parallel beamlets that travel in `self.direction`.
        """
        box_min = np.array([0, 0, 0])
        box_max = np.array(self.volume_shape)
        entry, exit = ray_box_intersection(self.source_point, self.direction, box_min, box_max)
        if entry is not None:
            plane_center = self.source_point + entry * self.direction
        else:
            plane_center = self.volume_center
        center = np.asarray(plane_center, dtype=float)

        ny, nx = self.raster_grid_size
        sy, sx = self.raster_spacing

        # Create orthonormal basis (u, v) for plane orthogonal to direction
        u, v = orthonormal_plane_basis(self.direction)

        y_idxs = (np.arange(ny) - (ny - 1) / 2.0)
        x_idxs = (np.arange(nx) - (nx - 1) / 2.0)
        # Offsets along basis
        x_offsets = x_idxs * sx
        y_offsets = y_idxs * sy
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")  # shape (ny, nx)

        # Broadcast into 3D coordinates
        pts = center.reshape((1, 1, 3)) + np.expand_dims(xx, axis=2) * u.reshape((1, 1, 3)) + np.expand_dims(yy, axis=2) * v.reshape((1, 1, 3))
        # Flatten into list of points [N,3]
        pts = pts.reshape((-1, 3))
        return pts

    def apply_dose(self, volume, intensities, energy_u, num_raster_points, depth_margin=0.5):
        """
        Deliver one energy layer across the raster points.
        (Now optimized with Numba)
        """
        # Use float64 for numba compatibility
        dose = np.zeros_like(volume, dtype=np.float64)
        box_min = np.array([0, 0, 0], dtype=np.float64)
        box_max = np.array(self.volume_shape, dtype=np.float64)
        beam_dir = self.direction.astype(np.float64)
        volume_length = float(self.volume_shape[0])
        grid_shape_array = np.array(self.volume_shape, dtype=np.int32)

        # Loop over each raster point 'p0', which is the start of a beamlet
        for i, p0 in enumerate(self.raster_points):
            p0_f64 = p0.astype(np.float64)
            
            # Find entry/exit (parametric depths along the ray from p0)
            intersection_times = ray_box_intersection(p0_f64, beam_dir, box_min, box_max)
            if intersection_times is None:
                continue
            t_entry_volume, t_exit_volume = intersection_times

            d_min = t_entry_volume + depth_margin
            d_max = t_exit_volume  - depth_margin
            if d_max <= d_min:
                continue
            
            energy_u_clamped = np.clip(energy_u, 0.0, 1.0)
            
            # Loop over energy layers
            for j, energy in enumerate(energy_u_clamped):
              peak_depth = d_min + energy * (d_max - d_min)
              
              intensity_idx = j * num_raster_points + i
              if intensity_idx >= len(intensities):
                  continue
              current_intensity = intensities[intensity_idx]
              
              if current_intensity == 0: # Optimization
                  continue
              
              # --- CALL THE NEW NUMBA HELPER ---
              _numba_voxel_loop(
                  dose,              # The array to modify
                  p0_f64,
                  beam_dir,
                  grid_shape_array,
                  t_entry_volume,
                  t_exit_volume,
                  current_intensity,
                  peak_depth,
                  volume_length
              )
            
        return dose, self.raster_points, self.source_point


# -----------------------
# Gymnasium environment
# -----------------------
class BeamAngleEnv(gym.Env):
    """
    Sequential, multi-beam radiotherapy environment.
    - Agent adds ONE beam per step.
    - max_steps = max_beams
    - Observation: [accumulated_dose_map, volume_mask]
    - Action: [gantry, couch, energies(L), intensities(L*N)] for a *single* beam.
    """
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.volume_shape = self.config["volume_shape"]
        self.raster_grid =  self.config["raster_grid"]
        self.raster_spacing =  self.config["raster_spacing"]
        self.voi_configs = self.config["voi_configs"]
        self.max_beams = self.config["max_beams"]
        self.max_steps = self.config["max_steps"] # Should be == max_beams
        self.target_size = self.config["target_size"]
        self.base_scene = self.config["base_config"]
        self.dose_target = self.config["dose_target"]
        self.epsilon = self.config["epsilon"]
        self.num_layers = self.config["num_layers"]
        
        # Number of raster points per beam
        self.num_raster_points = self.raster_grid[0] * self.raster_grid[1]
        self.volume_center = tuple(np.array(self.volume_shape) // 2)
        
        # Will be built in reset(), here is just to initialize
        self.volume_mask = np.zeros(self.volume_shape, dtype=np.float32)
        self.target_center = tuple(np.array(self.volume_shape) // 2)
        self.target_mask = np.zeros(self.volume_shape, dtype=bool)
        self.weights = []
        
        # --- SEQUENTIAL ACTION SPACE (FOR ONE BEAM) ---
        # Action: [gantry, couch, energies(L), intensities(L*N)]
        single_beam_low = np.concatenate([
            np.array([-np.pi, -np.pi/2], dtype=np.float32), # gantry, couch
            np.zeros(self.num_layers, dtype=np.float32),             # energies
            np.zeros(self.num_layers * self.num_raster_points, dtype=np.float32) # intensities
        ])
        single_beam_high = np.concatenate([
            np.array([np.pi, np.pi/2], dtype=np.float32), # gantry, couch
            np.ones(self.num_layers, dtype=np.float32),            # energies
            np.ones(self.num_layers * self.num_raster_points, dtype=np.float32) # intensities
        ])
        
        # The action space is just the single beam block
        self.action_space = spaces.Box(low=single_beam_low, high=single_beam_high, dtype=np.float32)
        self.action_dim = single_beam_low.size
        self.beam_params_shape = (self.max_beams*self.action_dim,)
        # --- END ACTION SPACE ---


        self.observation_space = spaces.Dict(
            {
                # The 3D dose map and mask
                "dose": spaces.Box(
                    low=0.0, 
                    high=np.finfo(np.float32).max,
                    shape=(2,) + self.volume_shape, 
                    dtype=np.float32
                ),
                # A 1D array holding all beam parameters (past and future)
                "beams": spaces.Box(
                    low=-np.pi, # Use a reasonable low (gantry angle)
                    high=np.pi,  # Use a reasonable high (gantry angle)
                    shape=self.beam_params_shape,
                    dtype=np.float32
                )
            }
        )
        self.beam_slots = [None for _ in range(self.max_beams)]
        self.step_count = 0
        self.dose_total = np.zeros(self.volume_shape, dtype=np.float32)
        self.beam_params_array = np.zeros(self.beam_params_shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None, config_override=None):
        super().reset(seed=seed)
        self.step_count = 0
        if config_override is not None:
            cfg = config_override
        else:
            cfg = perturb_config(self.base_scene, self.volume_shape, rng=self.np_random)

        # rebuild masks
        self.target_center = tuple(map(int, cfg["target_center"]))
        self.target_size = tuple(map(int, cfg["target_size"]))
        self.voi_configs = cfg["vois"]
        self.volume_mask = np.zeros(self.volume_shape, dtype=np.float32)
        self.weights = []
        for voi in self.voi_configs:
            center, weight, size = voi
            self._create_voi_mask(self.volume_mask, center, size, weight)
            self.weights.append(weight)
        self.target_mask = self._create_target_mask(self.volume_mask, self.target_center, self.target_size)

        # reset beam slots and accumulated dose
        self.beam_slots = [None for _ in range(self.max_beams)]
        self.dose_total = np.zeros(self.volume_shape, dtype=np.float32)
        self.beam_params_array = np.zeros(self.beam_params_shape, dtype=np.float32)
        self.state = np.stack([self.dose_total, self.volume_mask.astype(np.float32)], axis=0)
        obs = {
            "dose": self.state,
            "beams": self.beam_params_array
        }
        info = {}
        return obs, info

    
    def step(self, action):
        """
        Action is for ONE beam.
        We add this beam to the total dose.
        """
        action = np.asarray(action, dtype=float).ravel()
        
        # --- SEQUENTIAL LOGIC ---
        # The action is for the beam at the *current step*
        i = self.step_count # 0, 1, or 2
        start_idx = i * self.action_dim
        end_idx = (i + 1) * self.action_dim
        # Store the action
        self.beam_params_array[start_idx:end_idx] = action

        # 1. Parse the single-beam action
        block_size = 2 + self.num_layers + self.num_layers * self.num_raster_points
        if action.size != block_size:
            raise ValueError(f"Action length {action.size} != expected {block_size}")

        gantry = float(action[0])
        couch = float(action[1])
        energy_u = action[2: 2 + self.num_layers]
        ints = action[2 + self.num_layers : 2 + self.num_layers + self.num_layers * self.num_raster_points]
        
        # 2. Create and apply the beam
        beam = Beam(
            gantry_angle=gantry,
            couch_angle=couch,
            raster_grid_size=self.raster_grid,
            raster_spacing=self.raster_spacing,
            volume_shape=self.volume_shape
        )

        ints = np.clip(ints, 0.0, 1.0)
        direction = beam.direction
        dose_beam, raster_points, source_point = beam.apply_dose(
            np.zeros(self.volume_shape, dtype=np.float32),
            intensities=ints,
            energy_u=energy_u, num_raster_points=self.num_raster_points
        )
        
        # 3. ACCUMULATE the dose
        self.dose_total += dose_beam
        
        # 4. Store the beam
        self.beam_slots[i] = {
            "active": True, # It's now always active
            "gantry": gantry,
            "couch": couch,
            "intensities": ints.copy(),
            "raster_points": raster_points,
            "source_point": source_point,
            "direction": direction
        }
        # --- END SEQUENTIAL LOGIC ---

        # update state & compute reward
        self.state = np.stack([self.dose_total, self.volume_mask.astype(np.float32)], axis=0)
        obs = {
            "dose": self.state,
            "beams": self.beam_params_array
        }
        
        # Use the "dense" reward function that penalizes inaction
        reward = self._compute_reward(self.dose_total)

        self.step_count += 1
        
        # Terminated is now true ONLY on the last step
        terminated = (self.step_count >= self.max_steps)
        truncated = False
        info = {
            "step_count": self.step_count,
            "reward": reward,
            "beam_slots": self.beam_slots
        }
        return obs, float(reward), terminated, truncated, info


    # --------------------
    # Reward: DENSE + Inaction Penalty
    # --------------------
    def _compute_reward(self, dose):
        """
        Dense reward based on mean target dose, with a penalty for inaction
        and a penalty for OAR overdose.
        This logic is based on the user's successful single-beam code.
        """
        target_mask = np.isclose(self.volume_mask,1)
        if not np.any(target_mask):
             return -10.0 # Should not happen, but safeguard
        target_doses = dose[target_mask]
        
        # 1. DENSE REWARD: Reward the mean dose on the target.
        mean_target_dose = float(np.mean(target_doses)) if float(np.mean(target_doses))<1.0 else 1.0
        
        # 2. PENALTY FOR INACTION (and underdosing)
        # (1.0 - mean_target_dose) is a penalty that shrinks as mean dose approaches 1.0
        # If mean_target_dose = 0, target_penalty = 1.0
        target_penalty = 1.0 - mean_target_dose

        # 3. OAR PENALTY
        unique_vals = np.unique(self.volume_mask)
        oar_penalty = 0.0
        for val in unique_vals:
            if val == 0.0 or np.isclose(val, 1.0):
                continue
            voi_mask = np.isclose(self.volume_mask, val)
            if not np.any(voi_mask):
                continue
            mean_voi = float(np.mean(dose[voi_mask]))
            tolerance = float(val)
            if mean_voi > tolerance:
                # Penalty is quadratic above the tolerance
                oar_penalty += (mean_voi - tolerance)**2 

        # --- NEW REWARD STRUCTURE ---
        # Weights
        alpha = 1.0   # Weight for the target underdosing penalty
        beta = 1.5    # Weight for the OAR penalty
        
        # Logic: Reward = (mean dose) - (underdose penalty) - (OAR penalty)
        # - If "do nothing": reward = 0.0 - (1.0 * 1.0) - 0.0 = -1.0
        # - If "hit target" (mean=0.2): reward = 0.2 - (1.0 * (1.0-0.2)) - 0.0 = 0.2 - 0.8 = -0.6
        # Since -0.6 > -1.0, the agent is forced to act.
        reward =  - alpha * target_penalty - beta * oar_penalty
        
        reward = float(np.clip(reward, -10.0, 10.0))
        return reward
    

    @staticmethod
    def _create_target_mask(mask_volume, target_center, target_size):
        mask = mask_volume
        x0, y0, z0 = target_center
        dx, dy, dz = target_size
        if dx > 1:
            x_start = max(0, x0 - dx // 2)
            x_end = min(mask_volume.shape[0], x0 + dx // 2 + 1)
        else:
            x_start = x0
            x_end = x0 + 1
        if dy > 1:
            y_start = max(0, y0 - dy // 2)
            y_end = min(mask_volume.shape[1], y0 + dy // 2 + 1)
        else:
            y_start = y0
            y_end = y0 + 1
        if dz > 1:
            z_start = max(0, z0 - dz // 2)
            z_end = min(mask_volume.shape[2], z0 + dz // 2 + 1)
        else:
            z_start = z0
            z_end = z0 + 1
        # Set target mask value to 1
        mask[x_start:x_end, y_start:y_end, z_start:z_end] = 1
        return np.isclose(mask, 1) # Return a boolean mask

    @staticmethod
    def _create_voi_mask(mask_volume, voi_center, voi_size, weight):
        mask = mask_volume
        x0, y0, z0 = voi_center
        sx, sy, sz = voi_size
        if sx > 1:
            x_start = max(0, x0 - sx // 2)
            x_end = min(mask_volume.shape[0], x0 + sx // 2 + 1)
        else:
            x_start = x0
            x_end = x0 + 1
        if sy > 1:
            y_start = max(0, y0 - sy // 2)
            y_end = min(mask_volume.shape[1], y0 + sy // 2 + 1)
        else:
            y_start = y0
            y_end = y0 + 1
        if sz > 1:
            z_start = max(0, z0 - sz // 2)
            z_end = min(mask_volume.shape[2], z0 + sz // 2 + 1)
        else:
            z_start = z0
            z_end = z0 + 1
        # Set OAR mask value to its weight, only if not part of target
        existing_val = mask[x_start:x_end, y_start:y_end, z_start:z_end]
        mask[x_start:x_end, y_start:y_end, z_start:z_end][existing_val == 0] = float(weight)
        return mask

    def render(self, dose_threshold=0.05, beam_length=25, show=True, save_path=None):
        """
        Render a simplified 3D view:
        - Dose heatmap (translucent voxels)
        - One beam direction line per active beam
        - Red X markers for beam sources
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-5, self.state[0].shape[0]+5)
        ax.set_ylim(-5, self.state[0].shape[1]+5)
        ax.set_zlim(-5, self.state[0].shape[2]+5)
        ax.set_box_aspect([1, 1, 1])

    
        # Dose colormap
        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        # --- Base dose cloud (all voxels) ---
        x, y, z = np.where(self.state[0] > 0)
        dose_vals = self.state[0][x, y, z]
        ax.scatter(
            x+0.5, y+0.5, z+0.5,
            c=cmap(norm(dose_vals)),
            alpha=0.1,
            s=10,
            zorder=1
        )

        # --- Target voxels (big spheres) ---
        x_t, y_t, z_t = np.where(self.volume_mask == 1)
        dose_vals_t = self.state[0][x_t, y_t, z_t]
        ax.scatter(
            x_t + 0.5, y_t + 0.5, z_t + 0.5,
            c=cmap(norm(dose_vals_t)),
            marker='o',
            s=15,
            edgecolors='black',
            linewidths=1.5,
            alpha=1.0,
            zorder=3
        )

        # --- VOIs voxels (big cubes) ---
        for idx, weight in enumerate(self.weights):
            if weight == 1:  # skip target
                continue
            x_v, y_v, z_v = np.where(np.isclose(self.volume_mask,weight))
           
            dose_vals_v = self.state[0][x_v, y_v, z_v]
            ax.scatter(
                x_v + 0.5, y_v + 0.5, z_v + 0.5,
                c=cmap(norm(dose_vals_v)),
                marker='s',  # square marker = cube-like
                s=15,
                edgecolors='blue',
                linewidths=1.5,
                alpha=1.0,
                zorder=4
            )

        # --- Beam lines ---
        t = np.linspace(0, beam_length, 20)
        for beam in self.beam_slots:
            if beam is None: # No 'active' check needed
                continue
            src = np.array(beam["source_point"], dtype=float)
            ax.scatter(*src, color='red', marker='x', s=50, linewidths=2, zorder=5)

            for p0 in beam["raster_points"]:
                direction = np.array(beam["direction"], dtype=float)
                line = p0.reshape((3,1)) + np.outer(direction, t)
                ax.plot(line[0,:], line[1,:], line[2,:], color='cyan',  alpha=0.4 , linewidth = 0.8, zorder=2)

    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Dose Heatmap + Beam Directions")


        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(
            mappable,
            ax=ax,
            fraction=0.03,
            pad=0.1,
            label="Dose"
        )

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        return (img * 255).astype(np.uint8)

# --- UPDATED GIF CALLBACK ---
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
            out_dir = script_dir / "gifs_128"
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
# --- 128^3 VOLUME CONFIG ---
    
    # Define the 14 VOIs for the complex 128^3 scene
    # These are in two rings: an inner ring of 6, and an outer ring of 8
    # --- 64^3 VOLUME CONFIG ---
    
    # Define 6 VOIs for the 64^3 scene (a "cross" around the target)
    voi_configs_64 = [
        ((32, 32, 44), 0.5, (8, 8, 8)),  # Front
        ((32, 32, 20), 0.6, (8, 8, 8)),  # Back
        ((32, 44, 32), 0.4, (8, 8, 8)),  # Top
        ((32, 20, 32), 0.7, (8, 8, 8)),  # Bottom
        ((44, 32, 32), 0.5, (8, 8, 8)),  # Right
        ((20, 32, 32), 0.8, (8, 8, 8)),  # Left
    ]

    base_config_64 = {
        "volume_shape": (64, 64, 64),
        "target_center": (32, 32, 32),
        "target_size": (10, 10, 10), # Target size
        "vois": voi_configs_64,
    }

    config_env_128 =  {
        "volume_shape": (64, 64, 64),
        "target_size": (10, 10, 10),
        "base_config": base_config_64,
        "source_distance": 32,  # Scaled to new volume (64 // 2)
        "voi_configs": voi_configs_64,
        "epsilon": 1e-3,
        "dose_target": 1.0,
        "max_beams": 3,
        "num_layers": 6,
        "raster_grid": (8, 8),        # <-- INCREASED: 8x8 beamlets
        "raster_spacing": (1.5, 1.5), # <-- INCREASED: Spacing
        "max_steps": 3, 
    }
    
    voi_configs18 = [
        ((6, 6, 6), 0.5, (3, 3, 3)),   # left
        ((12, 12, 12), 0.8, (3, 3, 3)),  # right
        ((9, 4, 9), 0.6, (3, 3, 3)),   # down
        ((9, 15, 9), 0.4, (3, 3, 3)),  # up
        ((9, 9, 3), 0.7, (3, 3, 3)),   # back
        ((9, 9, 15), 0.9, (3, 3, 3)),  # front
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
        # --- SEQUENTIAL CONFIG ---
        "max_steps": 3, # Must be equal to max_beams
    }
    
    
    # Test the sequential environment
    if test:
        print("--- RUNNING SEQUENTIAL TEST ---")
        env = BeamAngleEnv(config_env_128)
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
        
        # --- 1. DEFINE BATCH SIZES ---
        num_workers = 9  # Use 9 workers
        steps_per_episode = config_env_128["max_steps"] # 3
        
        # This is (9 workers * 1 episode * 3 steps/episode)
        total_batch_size = num_workers * steps_per_episode  # 27
        
        # This MUST be <= 27. 3 or 9 are good choices.
        mini_batch_size = 3 

        ray.init(object_store_memory=8 * 1024**3) # Give Ray 8GB of RAM

        ModelCatalog.register_custom_model(
             "CustomConv3DModel",
             CustomConv3DModel,
         )
        
        register_env("CustomEnv-v0", lambda config: BeamAngleEnv(config))
        
        # --- 2. THE FIX: DEFINE CONFIG AS A DICTIONARY ---
        # This setup bypasses the PPOConfig() validation bug
        
        config = {
            # --- Environment Config ---
            "env": "CustomEnv-v0",
            "env_config": config_env_128,
            
            # --- Framework & Resources ---
            "framework": "torch",
            "num_gpus": 1,
            
            # --- Worker Config ---
            "num_workers": num_workers,
            "num_cpus_per_env_runner": 1, # Correct key
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 1, 
            "sample_timeout_s": 600.0,
            
            # --- BATCH SIZE FIX (THIS WILL NOT BE IGNORED) ---
            #"train_batch_size": total_batch_size,
            #"sgd_minibatch_size": mini_batch_size, # CHECK THIS KEY
            
            # --- Model Config ---
            "model": {
                "custom_model": "CustomConv3DModel",
                "vf_share_layers": True,
            },
            
            # --- PPO-Specific Training Params ---
            "entropy_coeff_schedule": [
                [0, 0.2],
                [5000, 0.01],
            ],
            "kl_coeff": 0.3,
            "kl_target": 0.01,
            "use_kl_loss": True,
            "vf_clip_param": 10.0,
            "clip_param": 0.2,
            
            # --- Old API Stack (THE FIX) ---
            "enable_rl_module_and_learner": False,
            "enable_env_runner_and_connector_v2": False,
            
            # --- Callback ---
            "callbacks": GifCallback,
        }
        
        # --- 3. RUN THE TUNER ---
        tuner = tune.Tuner(
            "PPO", # Pass the algorithm string
            run_config=train.RunConfig(
                stop={"training_iteration":10000},
                checkpoint_config=train.CheckpointConfig(
                        checkpoint_at_end=True,
                        checkpoint_frequency=100,),
                verbose=2,
            ),
            param_space=config, # Pass the config dictionary
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
        env = BeamAngleEnv(config_env_128)
        
        # --- A new test config for 128^3 evaluation ---
        test_config = {
            "volume_shape": (128, 128, 128),
            "target_center": (60, 70, 64), # Shifted target
            "target_size": (10, 10, 10),
            "vois": [
                # Different locations and weights from the training config
                ((60, 70, 80), 0.5, (8, 8, 8)),    # Near target (front)
                ((60, 70, 48), 0.7, (8, 8, 8)),    # Near target (back)
                ((80, 70, 64), 0.6, (10, 10, 10)), # Right
                ((40, 70, 64), 0.8, (10, 10, 10)), # Left
                ((100, 100, 100), 0.4, (15, 15, 15)), # Outer corner
                ((30, 30, 30), 0.5, (15, 15, 15)),   # Outer corner
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