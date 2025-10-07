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
from customConvModel4D import CustomConv3DModel, CustomConv3DModelTF
import json
from io import BytesIO
import matplotlib.pyplot as plt
import imageio as iio
import torch
from pathlib import Path
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors



def ray_box_intersection(p0, d, box_min, box_max, eps=1e-9):
    """
    function to compute the intersection between the beam and the volume of interest
    returns the time of intersection
    t_entry_clipped: This is the "time" of intersection for the ray's entry point into the axis-aligned bounding box defined by box_min and box_max.
    The "time" here is a parameter along the ray, where a point p on the ray is defined as p = p0 + t * d. 
    A value of t greater than 0 means the point is in the direction of the ray's travel.
    t_entry_clipped is the first non-negative value of t where the ray enters the box. If the ray starts inside the box, t_entry_clipped will be 0.
    t_exit: This is the "time" of intersection for the ray's exit point from the bounding box. 
    It represents the value of t where the ray leaves the box.

    
    """
    tmin = np.zeros(3)
    tmax = np.zeros(3)

    for i in range(3):
        if abs(d[i]) < eps:
            # Ray is parallel to axis i
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return None  # Ray is outside the slab and never intersects
            else:
                # Ray inside slab along axis i
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
        return None  # No intersection

    t_entry_clipped = max(t_entry, 0)
    return t_entry_clipped, t_exit


def traverse_grid_ray(p0, d, grid_shape=(64, 64, 64)):
    """
    function that implements the Voxel Traversal Algorithm, for computing the voxels that are affected by the action of the beam
    returns a list of tuples (x,y,z), that are the voxels that are affected by the beam
    """
    d = np.array(d, dtype=float)# direction of the beam
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        raise ValueError("Direction vector cannot be zero")
    d /= d_norm

    box_min = np.array([0, 0, 0], dtype=float)
    box_max = np.array(grid_shape, dtype=float)
    if ray_box_intersection(np.array(p0, dtype=float), d, box_min, box_max) is not None:
      t_entry, t_exit = ray_box_intersection(np.array(p0, dtype=float), d, box_min, box_max)
    else:
      return None

    if t_entry is None:
        return []  # The ray does not intersect the grid at all

    # Start from the entry point
    p = np.array(p0, dtype=float) + t_entry * d
    voxel = np.floor(p).astype(int)

    step = np.sign(d).astype(int)
    t_delta = np.empty(3)
    t_max = np.empty(3)

    result = []

    for i in range(3):
        if d[i] == 0:
            t_delta[i] = np.inf
            t_max[i] = np.inf
        else:
            next_boundary = voxel[i] + (step[i] > 0)
            t_max[i] = (next_boundary - p[i]) / d[i]
            t_delta[i] = 1 / abs(d[i])

    # Walk through the grid until we pass t_exit
    t = t_entry
    while (0 <= voxel[0] < grid_shape[0] and
           0 <= voxel[1] < grid_shape[1] and
           0 <= voxel[2] < grid_shape[2] and
           t <= t_exit):
        result.append(tuple(voxel.copy()))

        axis = np.argmin(t_max)
        t = t_max[axis]
        voxel[axis] += step[axis]
        t_max[axis] += t_delta[axis]

    return  result

def bragg_peak(distance, peak_depth, sigma=0.8, entrance=0.02, tail=0.01):
    """
    Toy depth-dose model along the beam ray:
    - small entrance dose before the peak,
    - Gaussian Bragg peak centered at `peak_depth`,
    - small distal tail after the peak.
    `distance` is the along-ray depth from the source (>= 0).
    """
    if distance < 0:
        return 0.0
    peak = np.exp(-0.5 * ((distance - peak_depth) / sigma) ** 2)
    return float(entrance + peak + tail * max(distance - peak_depth, 0.0))


# -------------------- For training using a perturbation of the original VOI, target configuration --------------------
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
        rng = np.random

    # perturb target center by integer offsets
    tcenter = np.array(base_config["target_center"], dtype=int)
    shift = rng.randint(-max_shift, max_shift+1, size=3)
    size = base_config["target_size"]
    new_tcenter = clamp_center(tcenter + shift, size, volume_shape)

    # VOIs: perturb each center a bit and also jitter weight slightly (optional)
    new_vois = []
    for (c, w, size) in base_config["vois"]:
        c = np.array(c, dtype=int)
        shift = rng.randint(-max_shift, max_shift+1, size=3)
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
     class for creating a Beam object that will be central in our final gym environment:
     The central ray of the beam begins from a point on a semisphere of radius equal to source_distance,
     and the position is specified by the two angles, the direction will be the the radial direction from the source.
     For now the raster scan is implemented as if the beam is made of parallel rays on a grid orthogonal to the ray direction.
    """

    def __init__(self, gantry_angle, couch_angle, raster_grid_size, raster_spacing, volume_shape, source_distance):
        self.gantry_angle = gantry_angle # the two angles that allow to specify the direction of the beam
        self.couch_angle = couch_angle
        self.raster_grid_size = raster_grid_size  #dimension of the grid of rays
        self.raster_spacing = raster_spacing  #spacing of the grid
        self.volume_shape = volume_shape #the shape of the volume of interest
        
        self.volume_center = np.array(volume_shape) /2.0 +0.5
        
        
        self.direction = self.get_direction()#each time a new beam object is instantiated the direction is computed
        self.source_distance = source_distance
        self.source_point = self.get_source_point()# "  "
        #self.bragg_peak_depth = np.linalg.norm(np.array(self.source_point) - self.volume_center )# I think it should coincide with the source distance, but for now I left it as different variable
          # distance of the source to the center of the volume

        self.raster_points = self.get_raster_points(center = np.array(volume_shape)//2) # "  "


    def get_direction(self):
        """
        Compute unit direction vector from gantry and couch angles.
        θ: gantry (rotation around z)
        φ: couch (rotation around x in patient coordinates)
        """
        θ = self.gantry_angle
        φ = self.couch_angle

        # Initially assume beam goes along z-axis (0, 0, 1)
        d = np.array([0, 0, 1])

        # Apply couch rotation (around x)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(φ), -np.sin(φ)],
            [0, np.sin(φ), np.cos(φ)],
        ])

        # Apply gantry rotation (around z)
        Rz = np.array([
            [np.cos(θ), -np.sin(θ), 0],
            [np.sin(θ), np.cos(θ), 0],
            [0, 0, 1],
        ])

        return (Rz @ Rx @ d)

    def get_raster_points(self, center=None ):
        """
        Create raster points in a 2D grid orthogonal to the beam direction.
        center specifies the center of the raster grid in the plane orthogonal to the beam direction
        """
        ny, nx = self.raster_grid_size
        sx, sy = self.raster_spacing

        # Raster grid centered on the source point, on plane orthogonal to beam direction
        # Find orthonormal basis (u, v) perpendicular to beam direction
        d = self.direction
        d_row = d.reshape(1, -1)
        basis = null_space(d_row)
        u = basis[:, 0]
        v = basis[:, 1]
        if center is None:
            center = self.source_point

        points = []
        for iy in range(-ny // 2, ny // 2 +1):
            for ix in range(-nx // 2, nx // 2 +1):
                offset =  ix * sx * u + iy * sy * v
                pt = center + offset + np.array([0.5, 0.5, 0.5])  # Center voxel coordinates, I add the last term so is centered
                points.append(pt)

        return points

    def get_source_point(self):
        """
        Compute the source point along -direction, since will depend on the gantry angle in general the position of the source
        """
        return self.volume_center - self.source_distance * self.direction # we compute the source point given the direction and the center position

    def apply_dose(self, volume, intensities, energy_u, num_raster_points, peak_sigma=0.8, depth_margin=0.5):
        """
        Deliver one energy layer across the raster points.
        energy_u in [0,1] selects the Bragg peak depth between the ray's entry/exit in the volume.
        """
        dose = np.zeros_like(volume, dtype=np.float32)

        for i, target in enumerate(self.raster_points):
            # Ray from source to raster point
            beam_vec = target - self.source_point
            beam_len = np.linalg.norm(beam_vec)
            if beam_len == 0:
                print("Warning: Zero length beam vector, skipping this raster point.")
                continue
            beam_dir = beam_vec / beam_len

            # Voxel traversal along this ray
            voxel_path = traverse_grid_ray(self.source_point, beam_dir, self.volume_shape)
            if voxel_path is None or len(voxel_path) == 0:
                continue

            # Find entry/exit (parametric depths along the ray from the source)
            box_min = np.array([0, 0, 0], dtype=float)
            box_max = np.array(self.volume_shape, dtype=float)
            intersection_times = ray_box_intersection(self.source_point, beam_dir, box_min, box_max)
            if intersection_times is None:
                continue
            t_entry_volume, t_exit_volume = intersection_times

            # Map normalized energy to a usable peak depth (stay a bit away from borders)
            d_min = t_entry_volume + depth_margin
            d_max = t_exit_volume  - depth_margin
            if d_max <= d_min:
                continue
            energy_u_clamped = np.clip(energy_u, 0.0, 1.0)
            peak_depth = d_min + energy_u_clamped * (d_max - d_min)

            # Deposit dose along this ray using along-ray depth
            for j, energy in enumerate(energy_u_clamped):
              for (x, y, z) in voxel_path:
                voxel_center = np.array([x, y, z]) + 0.5
                depth_along_beam = np.dot(voxel_center - self.source_point, beam_dir)
                dose[x, y, z] += intensities[num_raster_points*j + i] * bragg_peak(
                distance=depth_along_beam,
                peak_depth=peak_depth[j],
                sigma=peak_sigma,
            )

        return dose, self.raster_points, self.source_point







########################################################################################
#final environment that will be used for training the agent
##########################################################################################
import gymnasium as gym
from gymnasium import spaces
import numpy as np
class BeamAngleEnv(gym.Env):
    """
  The actual class that implements the final environmnet:
  -Uses the Beam Class for getting all the beam simulation for each step
  -gives the possibility to choose the configuration ( position and size of target tumor and VOI's),
   for VOIs  you need to give also the weight (susceptibility to radiation)
  -The aim is to optimize both the angle choice and the intensities of the different rays
  -For now just one beam, but is possible to add different rays with different directions
    """

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.volume_shape = self.config["volume_shape"]
        self.source_distance = self.config["source_distance"]
        self.raster_grid =  self.config["raster_grid"]
        self.raster_spacing =  self.config["raster_spacing"]
        self.voi_configs = self.config["voi_configs"] #is this needed?
        self.max_steps = self.config["max_steps"]
        self.target_size = self.config["target_size"]#dimension of the target
        self.base_scene = self.config["base_config"]#the base config that then we will use with some perturbation
        self.step_count = 0
        self.raster_points = []
        self.num_layers = 6 # layers of energy of the beams
        

        # Target mask (small cube at center)
        self.target_center = tuple(np.array(self.volume_shape) // 2)
        self.volume_mask = np.zeros(self.volume_shape, dtype=np.float32)# the mask that will contain both target and Voi's mask, 1 on target positions
                                                                        # and weight values on the VOIs positions
        self.target_mask = self._create_target_mask(
            self.volume_mask , self.target_center, self.target_size
        )# puts ones on the target position

        self.weights = []
        for voi in self.voi_configs:
            center, weight, voi_size = voi
            self._create_voi_mask(self.volume_mask, center, voi_size, weight)
            self.weights.append(weight)


        # Number of raster points (rays) per beam
        self.num_raster_points = (self.raster_grid[0]+1) * (self.raster_grid[1]+1)

        # Action space: [layers, gantry_angle, couch_angle, intensities...]
        
        low = np.concatenate([
            np.array([0.0,0.0,0.0,0.0,0.0,0.0, 0.0, -np.pi/2], dtype = np.float32),#energy and angle
            np.zeros(self.num_raster_points*self.num_layers, dtype= np.float32)#fluence for every beamlet 
        ])
        high = np.concatenate([
            np.array([1.0,1.0,1.0,1.0,1.0,1.0,2*np.pi, np.pi/2], dtype= np.float32),
            np.ones(self.num_raster_points*self.num_layers, dtype=np.float32)
        ])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Observation: current 3D dose map 
        self.observation_space = spaces.Box(low=0.0, high=np.finfo(np.float32).max,
    shape=(2,) + self.volume_shape, dtype=np.float32)





    def reset(self, *, seed=None, options=None, config_override = None):
        super().reset(seed=seed)
        self.step_count = 0
        if config_override is not None: #test case
          config = config_override
        else:
            config = perturb_config(self.base_scene, self.volume_shape)#otherwise we take the base config and we perturb it
        #recompute the key quantities of new configuration to be used in the next step
        self.target_center = tuple(map(int, config["target_center"]))
        self.target_size = tuple(map(int, config["target_size"]))
        self.voi_configs = config["vois"]
        self.weights = []
        # Reset volume mask
        self.volume_mask = np.zeros(self.volume_shape, dtype=np.float32)
        for voi in self.voi_configs:
            center, weight, voi_size = voi
            self._create_voi_mask(self.volume_mask, center, voi_size, weight)
            self.weights.append(weight)


        

        # Create target mask
        self.target_mask = self._create_target_mask(
            self.volume_mask, self.target_center, self.target_size
        )
        
        
        dose = np.zeros(self.volume_shape, dtype=np.float32)#we initialize the dose to zero volume
        self.state = np.stack([dose, self.volume_mask.astype(np.float32)], axis=0)
        obs = self.state
        info = {}
        return obs, info
        

    def step(self, action):
    
        energy_u   = action[0:self.num_layers]                 # ∈ [0,1], maps to depth
        self.gantry_angle     = float(action[self.num_layers])
        self.couch_angle      = float(action[self.num_layers+1])
        intensities = action[self.num_layers +2:]
        intensities = np.clip(intensities, 0, 1)

        #  Build beam and compute dose  
        
        volume_center = np.array(self.volume_shape) / 2.0
        beam = Beam(gantry_angle=self.gantry_angle,
            couch_angle=self.couch_angle,
            raster_grid_size=self.raster_grid,
            raster_spacing=self.raster_spacing,
            volume_shape=self.volume_shape,
            source_distance=self.source_distance)

        dose_beam, raster_points, source_point = beam.apply_dose(
            np.zeros(self.volume_shape, dtype=np.float32),
            intensities,
            energy_u=energy_u, num_raster_points=self.num_raster_points
        )
        self.raster_points = raster_points
        self.source_point = source_point

        self.state = np.stack([dose_beam, self.volume_mask.astype(np.float32)], axis=0)
        
        obs = self.state  
        reward = self._compute_multi_voi_reward(self.state[0])

        terminated = self.step_count >= self.max_steps
        truncated = False
        # Debug info
        self.step_count += 1
        info = {
            'step_count': self.step_count,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated
            
        }

      

        return obs, reward, terminated, truncated, info
        
    @staticmethod
    def _create_target_mask(mask_volume, target_center, target_size):
        mask = mask_volume
        x0, y0, z0 = target_center
        dx, dy, dz = target_size
        if dx>1:
          x_start = max(0, x0 - dx // 2)
          x_end = min(mask_volume.shape[0], x0 + dx // 2+1)
        else:
          x_start = x0
          x_end = x0+1
        if dy>1:
          y_start = max(0, y0 - dy // 2)
          y_end = min(mask_volume.shape[1], y0 + dy // 2)+1
        else:
          y_start = y0
          y_end = y0+1
        if dz>1:
          z_start = max(0, z0 - dz // 2)
          z_end = min(mask_volume.shape[2], z0 + dz // 2+1)
        else:
          z_start = z0
          z_end = z0+1
        mask[x_start:x_end, y_start:y_end, z_start:z_end] = 1


        return mask

    @staticmethod
    def _create_voi_mask(mask_volume, voi_center, voi_size, weight):
        mask = mask_volume
        x0, y0, z0 = voi_center
        sx, sy, sz = voi_size
        if sx>1:
          x_start = max(0, x0 - sx // 2)

          x_end = min(mask_volume.shape[0], x0 + sx // 2+1)
        else:
          x_start = x0
          x_end = x0+1
        if sy>1:
          y_start = max(0, y0 - sy // 2)
          y_end = min(mask_volume.shape[1], y0 + sy // 2+1)
        else:
          y_start = y0
          y_end = y0+1
        if sz>1:
          z_start = max(0, z0 - sz // 2)
          z_end = min(mask_volume.shape[2], z0 + sz // 2+1)
        else:
          z_start = z0
          z_end = z0+1


        mask[x_start:x_end, y_start:y_end, z_start:z_end] = weight
        return mask


    def _compute_multi_voi_reward(self, dose):
        """
        Reward penalizes both if the target is not covered by radiation, and if the VOI are too irradiated
        """
        self.target_mask = np.isclose(self.volume_mask,1)
        dose_target = dose[self.target_mask]
        mean_target = dose_target.mean() if dose_target.mean() <1  else 1.0
        penalty=0

        penalty -= (1-mean_target)
        for  w in self.weights:
            voi_mask = np.isclose(self.volume_mask, w)

            dose_voi = dose[voi_mask]
            mean_voi = dose_voi.mean() if dose_voi.size > 0 else 0.0
            if mean_voi>w:#the weight establish the level of tolerance 
               penalty -=  mean_voi
        
        
        
            

        return float(penalty)
    def render(self, mode='rgb_array'):
        """function to visualize the intensity of the radiation on the volume, target is identified with small black balls, VOI's with small blue cubes"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.state[0].shape[0])
        ax.set_ylim(0, self.state[0].shape[1])
        ax.set_zlim(0, self.state[0].shape[2])
        ax.set_box_aspect([1, 1, 1])  # cube proportions

        # Dose colormap
        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        # --- Base dose cloud (all voxels) ---
        x, y, z = np.where(self.state[0] > 0)
        dose_vals = self.state[0][x, y, z]
        ax.scatter(
            x+0.5, y+0.5, z+0.5,
            c=cmap(norm(dose_vals)),
            alpha=0.4,
            s=10
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

        # --- Rays ---
        t = np.linspace(0, 10, 20)
        for p0 in self.raster_points:
            d = p0 - self.source_point
            d /= np.linalg.norm(d)
            x = self.source_point[0] + d[0] * t
            y = self.source_point[1] + d[1] * t
            z = self.source_point[2] + d[2] * t
            ax.plot(
                x, y, z,
                color='gray',
                alpha=0.3,
                linewidth=0.8
            )

        # Add colorbar for dose reference
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

    def close(self):
        plt.close()








#for checking the situation during training
class GifCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        iteration = result["training_iteration"]

        if iteration % 100 == 0:
            env = BeamAngleEnv(config_env)
            obs, _ = env.reset()
            #frames = [env.render()]
            terminated = truncated = False

            for _ in range(1):
                action = algorithm.compute_single_action(obs, explore=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                frames.append(env.render())
                if terminated or truncated:
                    break
            

            script_dir = Path(__file__).resolve().parent
            out_dir = script_dir / "gifs"
            out_dir.mkdir(exist_ok=True)
            gif_path = out_dir / f"training_iter_{iteration}.gif"


            iio.mimsave(gif_path, frames, duration=0.3)
            print(f"Saved GIF at {gif_path}")

            env.close()



if __name__ == "__main__":
    test = False
    evaluation = False
    training = not test and not evaluation

    
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

    # Target
    "target_center": (9, 9, 9),
    "target_size": (3, 3, 3),

    # List of VOIs: (center, weight, size)
    "vois": [
        ((6, 6, 6), 0.5, (3, 3, 3)),   # left
        ((12, 12, 12), 0.8, (3, 3, 3)),  # right
        ((9, 4, 9), 0.6, (3, 3, 3)),   # down
        ((9, 15, 9), 0.4, (3, 3, 3)),  # up
        ((9, 9, 3), 0.7, (3, 3, 3)),   # back
        ((9, 9, 15), 0.9, (3, 3, 3)),  # front
    ]
}



    config_env =  {
    "volume_shape": (18, 18, 18),
    "target_size": (3, 3, 3),
    #"bragg_peak_depth": 9,
    "base_config": base_config,
    "source_distance": 9,
    "voi_configs": voi_configs18,
    "raster_grid": (4, 4),
    "raster_spacing": (1.0, 1.0),
    "max_steps": 1,
}
    
    
    # Test the environment
    if test:
        print("test not implemented yet")
       
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
        register_env("CustomEnv-v0", lambda config_env: BeamAngleEnv(config_env))
        # Define RLlib configuration for PPO
        config = (
            PPOConfig()
            .framework("torch")
            .environment("CustomEnv-v0", env_config=config_env)
            .callbacks(GifCallback)
            .resources(
                 num_gpus=1,
              )  # Use 1 GPU for training
            .env_runners(
                num_env_runners=31, 
                #num_gpus_per_env_runner=1,
                preprocessor_pref=None, # The default gave nan observations after some point
                #num_envs_per_env_runner=10,
                #batch_mode="complete_episodes",
                #rollout_fragment_length=80,
            )
            .training(
                model={
                    "custom_model": "CustomConv3DModel",
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
                stop={"training_iteration":10000},
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
        register_env("CustomEnv-v0", lambda config_env: BeamAngleEnv(config_env))

        ModelCatalog.register_custom_model(
            "CustomConv3DModel",
            CustomConv3DModel,
        )

        ModelCatalog.register_custom_model(
            "conv3d_model_tf",
              CustomConv3DModelTF
        )

    
        rl_module = Policy.from_checkpoint(
            path_to_policy
            )
        env = BeamAngleEnv(config_env)
        

        
        # For testing we use a configuration not used in the training but still close to the original
        test_config = {
    "target_center": (9, 9, 9),
    "target_size": (3, 3, 3),
    "vois": [
        # Six VOIs (all size 3x3x3), chosen near the center but not equal to your original 6.
        ((12, 12, 9), 0.65, (3, 3, 3)),  # diagonal up-right (x+, y+)
        ((6, 12, 9),  0.55, (3, 3, 3)),  # diagonal up-left  (x-, y+)
        ((12, 6, 9),  0.75, (3, 3, 3)),  # diagonal down-right (x+, y-)
        ((6, 6, 9),   0.45, (3, 3, 3)),  # diagonal down-left  (x-, y-)
        ((9, 12, 6),  0.70, (3, 3, 3)),  # up + back/front mix (y+, z-)
        ((12, 9, 6),  0.60, (3, 3, 3)),  # right + back mix (x+, z-)
    ]
}
    

        # Reset env with the new test config
        obs, _ = env.reset(config_override=test_config)

        frames = []
        terminated = False 
        frames.append(env.render())
        # Test step
        for _ in range(1):
            action = rl_module.compute_single_action(obs, explore=False)[0]
            obs, reward, terminated, truncated, info = env.step(action)

            frames.append(env.render())
        ''''''
        # Save the frames as a GIF
        print("Final step reward:", reward)
        iio.mimsave('eval.gif', frames, duration=1)

        env.close()
        ray.shutdown()



