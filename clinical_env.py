"""
Sequential, multi-beam radiotherapy environment for reinforcement learning.
OPTIMIZED VERSION: Uses Numba for fast dose calculation and includes collision detection.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import imageio as iio
from io import BytesIO
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- NUMBA IMPORTS ---
from numba import njit, prange

# -----------------------
# 1. NUMBA KERNELS (Fast Math)
# -----------------------

@njit(fastmath=True)
def ray_box_intersection_numba(p0, d, box_min, box_max):
    """
    Calculates t_entry and t_exit for a ray entering a box.
    Returns (-1.0, -1.0) if no intersection.
    """
    tmin = -1e30 # Representing -inf
    tmax = 1e30  # Representing +inf
    
    for i in range(3):
        if abs(d[i]) < 1e-9:
            # Ray parallel to axis
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return -1.0, -1.0
        else:
            inv_d = 1.0 / d[i]
            t1 = (box_min[i] - p0[i]) * inv_d
            t2 = (box_max[i] - p0[i]) * inv_d
            
            if t1 < t2:
                if t1 > tmin: tmin = t1
                if t2 < tmax: tmax = t2
            else:
                if t2 > tmin: tmin = t2
                if t1 < tmax: tmax = t1
                
    if tmin > tmax or tmax < 0:
        return -1.0, -1.0
        
    return max(tmin, 0.0), tmax

@njit(fastmath=True)
def bragg_peak_numba(depth, peak_depth, volume_length=18.0):
    # Hardcoded parameters for speed 
    width_pre = (volume_length * 6.0) / 250.0
    width_post = (volume_length * 1.8) / 250.0
    entrance_scale = 0.2
    tail_scale = 0.05
    
    if depth < 0.0: depth = 0.0
    
    val = 0.0
    diff = depth - peak_depth
    
    if depth <= peak_depth:
        # Rise component
        rise = np.exp(-(diff * diff) / (2.0 * width_pre * width_pre))
        val = entrance_scale + (rise * (1.0 - entrance_scale))
    else:
        # Fall component
        fall = np.exp(-(diff * diff) / (2.0 * width_post * width_post))
        tail = tail_scale * np.exp(-diff / (width_post * 1.5))
        val = (fall * (1.0 - tail_scale)) + tail

    return max(val, 0.0)

@njit(parallel=True, fastmath=True)
def compute_dose_numba(volume, raster_points, direction, intensities, energy_u, 
                       volume_shape, num_layers, depth_margin):
    """
    The Main Kernel. 
    Iterates over all raster points (in parallel), traces the grid, 
    and adds dose to 'volume' in-place.
    """
    nx, ny, nz = volume_shape
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([float(nx), float(ny), float(nz)])
    num_raster = len(raster_points)
    
    # Thread-local buffers to avoid race conditions
    num_threads = 8
    local_volumes = np.zeros((num_threads, nx, ny, nz), dtype=np.float32)
    
    # Parallel loop over thread chunks
    for t_id in prange(num_threads):
        chunk_size = (num_raster + num_threads - 1) // num_threads
        start_idx = t_id * chunk_size
        end_idx = min(start_idx + chunk_size, num_raster)
        
        for i in range(start_idx, end_idx):
            p0 = raster_points[i]
            
            # 1. Ray Box Intersection
            t_entry, t_exit = ray_box_intersection_numba(p0, direction, box_min, box_max)

            if t_entry < 0 and t_exit < 0:
                continue 
                
            d_min = t_entry + depth_margin
            d_max = t_exit - depth_margin

            if d_max <= d_min:
                continue
                
            # 2. Voxel Traversal Setup (Amanatides & Woo simplified)
            current_t = t_entry
            start_pos = p0 + current_t * direction
            
            vx = int(np.floor(start_pos[0]))
            vy = int(np.floor(start_pos[1]))
            vz = int(np.floor(start_pos[2]))
            
            # Handle boundary cases
            if vx == nx and direction[0] < 0: vx = nx - 1
            if vy == ny and direction[1] < 0: vy = ny - 1
            if vz == nz and direction[2] < 0: vz = nz - 1
            
            vx = max(0, min(vx, nx - 1))
            vy = max(0, min(vy, ny - 1))
            vz = max(0, min(vz, nz - 1))
            
            step_x = 1 if direction[0] >= 0 else -1
            step_y = 1 if direction[1] >= 0 else -1
            step_z = 1 if direction[2] >= 0 else -1
            
            # Calculate t_max (dist to next boundary)
            if direction[0] == 0:
                t_max_x = 1e30
                t_delta_x = 1e30
            else:
                t_delta_x = abs(1.0 / direction[0])
                next_bound_x = np.floor(start_pos[0]) + (1 if step_x > 0 else 0)
                t_max_x = (next_bound_x - p0[0]) / direction[0]

            if direction[1] == 0:
                t_max_y = 1e30
                t_delta_y = 1e30
            else:
                t_delta_y = abs(1.0 / direction[1])
                next_bound_y = np.floor(start_pos[1]) + (1 if step_y > 0 else 0)
                t_max_y = (next_bound_y - p0[1]) / direction[1]

            if direction[2] == 0:
                t_max_z = 1e30
                t_delta_z = 1e30
            else:
                t_delta_z = abs(1.0 / direction[2])
                next_bound_z = np.floor(start_pos[2]) + (1 if step_z > 0 else 0)
                t_max_z = (next_bound_z - p0[2]) / direction[2]

            # 3. Walk the grid
            while current_t < t_exit:
                # --- Deposit Dose ---
                cx = float(vx) + 0.5
                cy = float(vy) + 0.5
                cz = float(vz) + 0.5

                depth_along = (cx - p0[0])*direction[0] + (cy - p0[1])*direction[1] + (cz - p0[2])*direction[2]

                if depth_along >= t_entry - 1e-6:
                    for j in range(num_layers):
                        idx = j * num_raster + i
                        val_int = intensities[idx]
                        if val_int > 1e-4:
                            peak = d_min + energy_u[j] * (d_max - d_min)
                            bp_val = bragg_peak_numba(depth_along, peak, 18.0)
                            # Write to thread-local buffer (no race condition)
                            local_volumes[t_id, vx, vy, vz] += val_int * bp_val

                # --- Step ---
                if t_max_x < t_max_y:
                    if t_max_x < t_max_z:
                        vx += step_x
                        current_t = t_max_x
                        t_max_x += t_delta_x
                    else:
                        vz += step_z
                        current_t = t_max_z
                        t_max_z += t_delta_z
                else:
                    if t_max_y < t_max_z:
                        vy += step_y
                        current_t = t_max_y
                        t_max_y += t_delta_y
                    else:
                        vz += step_z
                        current_t = t_max_z
                        t_max_z += t_delta_z

                if vx < 0 or vx >= nx or vy < 0 or vy >= ny or vz < 0 or vz >= nz:
                    break
    
    # Reduction: sum thread-local buffers into main volume
    for t_id in range(num_threads):
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    volume[x, y, z] += local_volumes[t_id, x, y, z]

# -----------------------
# 2. UTILITY & GEOMETRY
# -----------------------

def orthonormal_plane_basis(v):
    v = np.asarray(v, dtype=float)
    if np.allclose(v, 0):
        raise ValueError("zero direction")
    if abs(v[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    u = a - v * np.dot(a, v)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(v, u)
    w = w / (np.linalg.norm(w) + 1e-12)
    return u, w

def clamp_center(center, size, volume_shape):
    clamped = []
    for i in range(3):
        half = size[i] // 2
        low = half
        high = volume_shape[i] - (size[i] - half)
        c = int(round(center[i]))
        c = max(low, min(high, c))
        clamped.append(c)
    return tuple(clamped)

def boxes_intersect(c1, s1, c2, s2):
    """
    Check if two 3D boxes intersect.
    c: center (x,y,z), s: size (w,h,d)
    Uses simple AABB logic (assuming axis aligned for safety check)
    """
    for i in range(3):
        # Calculate min and max for both boxes
        min1 = c1[i] - s1[i]//2
        max1 = min1 + s1[i]
        min2 = c2[i] - s2[i]//2
        max2 = min2 + s2[i]
        
        # If disjoint in any dimension, they don't intersect
        if max1 <= min2 or max2 <= min1:
            return False
    return True

def perturb_config(base_config, volume_shape, max_shift=1, rng=None):
    """
    Randomizes the scene configuration while preventing overlaps.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Place Target
    tcenter = np.array(base_config["target_center"], dtype=int)
    tsize = base_config["target_size"]
    shift = rng.integers(-max_shift, max_shift+1, size=3)
    new_tcenter = clamp_center(tcenter + shift, tsize, volume_shape)
    
    # 2. Place VOIs with Collision Check
    new_vois = []
    
    for (c, w, size) in base_config["vois"]:
        c = np.array(c, dtype=int)
        
        # Try to find a non-overlapping position (max 10 attempts)
        valid_pos = False
        final_c = c
        
        for _ in range(10): 
            shift = rng.integers(-max_shift, max_shift+1, size=3)
            candidate_c = clamp_center(c + shift, size, volume_shape)
            
            # Check collision with Target
            if boxes_intersect(new_tcenter, tsize, candidate_c, size):
                continue
            
            # Check collision with already placed VOIs
            collision_with_other = False
            for (existing_c, _, existing_size) in new_vois:
                if boxes_intersect(existing_c, existing_size, candidate_c, size):
                    collision_with_other = True
                    break
            if collision_with_other:
                continue

            valid_pos = True
            final_c = candidate_c
            break
        
        # If valid_pos is False, we use the last candidate but it might overlap.
        # Ideally, we fall back to original C or just accept it.
        # Here we use final_c which is the last attempted position.

        #w_new = float(np.clip(w + rng.normal(scale=0.05), 0.05, 10.0)) don't perturb constraints
        new_vois.append((final_c, w, size))

    return {
        "target_center": tuple(new_tcenter),
        "target_size": tuple(tsize),
        "vois": new_vois,
    }

# -----------------------
# 3. BEAM CLASS
# -----------------------
class Beam:
    def __init__(self, gantry_angle, couch_angle, raster_grid_size, raster_spacing, volume_shape):
        self.gantry_angle = float(gantry_angle)
        self.couch_angle = float(couch_angle)
        self.raster_grid_size = tuple(map(int, raster_grid_size))
        self.raster_spacing = tuple(map(float, raster_spacing))
        self.volume_shape = tuple(map(int, volume_shape))
        self.volume_center = np.array(self.volume_shape, dtype=float) // 2.0 + 0.5
        self.source_distance = self.volume_shape[0]//2 + 1
        self.direction = self._angles_to_direction(self.gantry_angle, self.couch_angle)
        self.source_point = self.get_source_point()
        self.raster_points = self.get_raster_points(center=self.source_point)

    def get_source_point(self):
        return self.volume_center - self.source_distance * self.direction
    
    def _angles_to_direction(self, gantry, elevation): 
        """
        gantry: azimuth in xy-plane [-π, π]
        elevation: angle from xy-plane [-π/2, π/2]
    
        elevation = 0 → horizontal beam (in xy-plane)
        elevation = π/2 → beam pointing up
        elevation = -π/2 → beam pointing down
        """
        x = np.cos(elevation) * np.cos(gantry) 
        y = np.cos(elevation) * np.sin(gantry) 
        z = np.sin(elevation) 
        v = np.array([x, y, z], dtype=float) 
        v /= np.linalg.norm(v) + 1e-12
        return -v
    
    def get_raster_points(self, center=None):
        # Re-using the numba intersection for setup is fine, or simple math
        # We'll stick to python here as it runs once per step
        box_min = np.array([0, 0, 0], dtype=float)
        box_max = np.array(self.volume_shape, dtype=float)
        
        # We can call the numba version here too, just need to handle tuple return
        t_entry, _ = ray_box_intersection_numba(self.source_point, self.direction, box_min, box_max)
        
        if t_entry >= 0:
            plane_center = self.source_point + t_entry * self.direction
        else:
            plane_center = self.volume_center
        center = np.asarray(plane_center, dtype=float)

        ny, nx = self.raster_grid_size
        sy, sx = self.raster_spacing

        u, v = orthonormal_plane_basis(self.direction)

        y_idxs = (np.arange(ny) - (ny - 1) / 2.0)
        x_idxs = (np.arange(nx) - (nx - 1) / 2.0)
        x_offsets = x_idxs * sx
        y_offsets = y_idxs * sy
        xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")

        pts = center.reshape((1, 1, 3)) + np.expand_dims(xx, axis=2) * u.reshape((1, 1, 3)) + np.expand_dims(yy, axis=2) * v.reshape((1, 1, 3))
        pts = pts.reshape((-1, 3))
        return pts

    def apply_dose(self, volume, intensities, energy_u, num_raster_points, depth_margin=0.5):
        """
        Calls the Numba kernel for high-speed dose deposition.
        """
        # Ensure contiguous arrays and correct types for C-level speed
        raster_pts_arr = np.ascontiguousarray(self.raster_points, dtype=np.float64)
        dir_arr = np.ascontiguousarray(self.direction, dtype=np.float64)
        ints_arr = np.ascontiguousarray(intensities, dtype=np.float64)
        energies_arr = np.ascontiguousarray(energy_u, dtype=np.float64)
        shape_arr = np.array(self.volume_shape, dtype=np.int64)
        
        # Numba function modifies 'volume' in-place
        compute_dose_numba(
            volume, 
            raster_pts_arr, 
            dir_arr, 
            ints_arr, 
            energies_arr, 
            shape_arr,
            len(energy_u),
            float(depth_margin)
        )
            
        return volume, self.raster_points, self.source_point

# -----------------------
# 4. GYM ENVIRONMENT
# -----------------------
class BeamAngleEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.volume_shape = self.config["volume_shape"]
        self.raster_grid =  self.config["raster_grid"]
        self.raster_spacing =  self.config["raster_spacing"]
        self.voi_configs = self.config["voi_configs"]
        self.max_beams = self.config["max_beams"]
        self.max_steps = self.config["max_steps"] 
        self.target_size = self.config["target_size"]
        self.base_scene = self.config["base_config"]
        self.dose_target = self.config["dose_target"]
        self.epsilon = self.config["epsilon"]
        self.num_layers = self.config["num_layers"]
        
        self.num_raster_points = self.raster_grid[0] * self.raster_grid[1]
        self.volume_center = tuple(np.array(self.volume_shape) // 2)
        
        self.volume_mask = np.zeros(self.volume_shape, dtype=np.float32)
        self.target_center = tuple(np.array(self.volume_shape) // 2)
        self.target_mask = np.zeros(self.volume_shape, dtype=bool)
        self.weights = []
        
        # Action space
        single_beam_low = np.concatenate([
            np.array([-np.pi, -np.pi/3], dtype=np.float32), 
            np.zeros(self.num_layers, dtype=np.float32),            
            np.zeros(self.num_layers * self.num_raster_points, dtype=np.float32) 
        ])
        single_beam_high = np.concatenate([
            np.array([np.pi, np.pi/3], dtype=np.float32), 
            np.ones(self.num_layers, dtype=np.float32),            
            np.ones(self.num_layers * self.num_raster_points, dtype=np.float32) 
        ])
        
        self.action_space = spaces.Box(low=single_beam_low, high=single_beam_high, dtype=np.float32)
        self.action_dim = single_beam_low.size
        #self.beam_params_shape = (self.max_beams*self.action_dim,)
        self.observation_space = spaces.Box(
            low=0.0, high=np.finfo(np.float32).max,
            shape=(2,) + self.volume_shape, dtype=np.float32
            )

        self.beam_slots = [None for _ in range(self.max_beams)]
        self.step_count = 0
        self.dose_total = np.zeros(self.volume_shape, dtype=np.float32)
        #self.beam_params_array = np.zeros(self.beam_params_shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None, config_override=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Use perturb_config with collision detection
        if config_override is not None:
            cfg = config_override
        else:
            cfg = perturb_config(self.base_scene, self.volume_shape, rng=self.np_random)
            #cfg = self.base_scene
        # Rebuild masks
        self.target_center = tuple(map(int, cfg["target_center"]))
        self.target_size = tuple(map(int, cfg["target_size"]))
        self.voi_configs = cfg["vois"]
        
        self.volume_mask = np.zeros(self.volume_shape, dtype=np.float32)
        self.weights = []
        
        # Create masks from config
        for voi in self.voi_configs:
            center, weight, size = voi
            self._create_voi_mask(self.volume_mask, center, size, weight)
            self.weights.append(weight)
            
        self.target_mask = self._create_target_mask(self.volume_mask, self.target_center, self.target_size)

        self.beam_slots = [None for _ in range(self.max_beams)]
        self.dose_total = np.zeros(self.volume_shape, dtype=np.float32)
        self.prev_target_cost = self._get_target_cost(self.dose_total)
        self.prev_oar_cost = self._get_oar_cost(self.dose_total)
        self.prev_body_cost = self._get_body_cost(self.dose_total)
        #self.beam_params_array = np.zeros(self.beam_params_shape, dtype=np.float32)
        
        self.state = np.stack([self.dose_total, self.volume_mask.astype(np.float32)], axis=0)
        obs = self.state
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=float).ravel()
        
        i = self.step_count 
        start_idx = i * self.action_dim
        end_idx = (i + 1) * self.action_dim
        #self.beam_params_array[start_idx:end_idx] = action

        gantry = float(action[0])
        couch = float(action[1])
        #print(f"Step {self.step_count}: gantry={gantry:.4f}, couch={couch:.4f}", flush=True)
        energy_u = action[2: 2 + self.num_layers]
        ints = action[2 + self.num_layers : 2 + self.num_layers + self.num_layers * self.num_raster_points]
        
        beam = Beam(
            gantry_angle=gantry,
            couch_angle=couch,
            raster_grid_size=self.raster_grid,
            raster_spacing=self.raster_spacing,
            volume_shape=self.volume_shape
        )

        ints = np.clip(ints, 0.0, 1.0)
        
        # This now calls the Numba-optimized version
        dose_beam, raster_points, source_point = beam.apply_dose(
            np.zeros(self.volume_shape, dtype=np.float32),
            intensities=ints,
            energy_u=energy_u, 
            num_raster_points=self.num_raster_points
        )
        
        self.dose_total += dose_beam
        
        self.beam_slots[i] = {
            "active": True,
            "gantry": gantry,
            "couch": couch,
            "intensities": ints.copy(),
            "raster_points": raster_points,
            "source_point": source_point,
            "direction": beam.direction
        }

        self.state = np.stack([self.dose_total, self.volume_mask.astype(np.float32)], axis=0)
        obs = self.state
        
        #reward = self._compute_reward(self.dose_total)
        # ---------------- NEW REWARD LOGIC ----------------
        # 1. Calculate New Costs
        curr_target_cost = self._get_target_cost(self.dose_total)
        curr_oar_cost = self._get_oar_cost(self.dose_total)
        #curr_body_cost = self._get_body_cost(self.dose_total)
        # 2. Calculate Diffs
        # Improvement > 0 if we added dose to tumor
        target_improvement = self.prev_target_cost - curr_target_cost
        
        # Degradation > 0 if we hit an OAR (Cost went up)
        oar_degradation = curr_oar_cost - self.prev_oar_cost
        #body_degradation = curr_body_cost - self.prev_body_cost
        # 3. Define Weights
        w_progress = 2  # Reward for fixing the tumor
        w_safety = 2   # Penalty for hitting OAR (Higher priority)
        w_final = 2 
        w_body = 0.1
        #penalty for hitting body volume
        
        # 4. Step Reward
        # Reward for target progress - Penalty for OAR damage
        reward = (w_progress * target_improvement) - (w_safety * oar_degradation) #- (w_body)

        self.step_count += 1
        terminated = (self.step_count >= self.max_steps)
        if terminated:
            reward -= (w_final * curr_target_cost)
            
        # 6. Update History
        self.prev_target_cost = curr_target_cost
        self.prev_oar_cost = curr_oar_cost
        
        # Clip for stability
        #reward = float(np.clip(reward, -20.0, 5.0))
        
        return obs, float(reward), terminated, False, {}

    def _get_target_cost(self, dose):
        """ Calculate Mean Squared Underdose Error for Target """
        target_mask = np.isclose(self.volume_mask, 1)
        if not np.any(target_mask): return 1.0 # Should not happen
        
        target_doses = dose[target_mask]
        # Error = (1.0 - dose)^2, only if dose < 1.0
        diff = 1.0 - target_doses
        underdose = np.maximum(diff, 0.0)
        return np.mean(underdose)

    def _get_oar_cost(self, dose):
        """ Calculate Mean Squared Overdose Error for OARs """
        total_oar_cost = 0.0
        unique_vals = np.unique(self.volume_mask)
        
        for val in unique_vals:
            if val == 0.0 or np.isclose(val, 1.0): continue
            
            voi_mask = np.isclose(self.volume_mask, val)
            if not np.any(voi_mask): continue
            
            voi_doses = dose[voi_mask]
            tolerance = float(val)
            
            # Error = (dose - tolerance)^2, only if dose > tolerance
            diff = voi_doses - tolerance
            overdose = np.maximum(diff, 0.0)
            total_oar_cost += np.mean(overdose)
            
        return total_oar_cost
    
    def _get_body_cost(self, dose):
        """ Calculate Mean Squared Overdose Error for Body Volume """
        body_mask = np.isclose(self.volume_mask, 0.0)
        if not np.any(body_mask): return 0.0 # Should not happen
        
        body_doses = dose[body_mask]
        tolerance = 0.0
        
        # Error = (dose - tolerance)^2, only if dose > tolerance
        diff = body_doses - tolerance
        overdose = np.maximum(diff, 0.0)
        return np.mean(overdose**2)

    def _compute_reward(self, dose):
        """
        Updated Reward using Voxel-Wise One-Sided Penalty.
        - Target: Penalizes (1.0 - dose)^2 only for voxels < 1.0. 
                  (Ignores overdose, penalizes cold spots).
        - OAR: Penalizes (dose - tolerance)^2 only for voxels > tolerance.
                  (Penalizes hot spots).
        """
        # 1. Check Target Existence
        target_mask = np.isclose(self.volume_mask, 1)
        if not np.any(target_mask):
             return -10.0 
        
        # --- TARGET PENALTY (Voxel-wise Underdose) ---
        target_doses = dose[target_mask]
        
        # Calculate difference: 1.0 - dose
        # If dose is 0.8, diff is 0.2 (Penalty).
        # If dose is 1.5, diff is -0.5. We clip this to 0.0 (No penalty for overdose).
        diff_target = 1.0 - target_doses
        underdose_errors = np.maximum(diff_target, 0.0)
        
        # Mean Squared Error of the underdosed voxels
        # This penalizes both low mean AND poor coverage (cold spots)
        target_penalty = np.mean(underdose_errors**2)
        
        # --- OAR PENALTY (Voxel-wise Overdose) ---
        unique_vals = np.unique(self.volume_mask)
        oar_penalty = 0.0
        
        for val in unique_vals:
            # Skip background (0) and target (1)
            if val == 0.0 or np.isclose(val, 1.0):
                continue
            
            voi_mask = np.isclose(self.volume_mask, val)
            if not np.any(voi_mask):
                continue
            
            voi_doses = dose[voi_mask]
            tolerance = float(val) # Using the mask value as the tolerance
            
            # Calculate excess: dose - tolerance
            # If dose > tolerance, we get positive penalty.
            # If dose < tolerance, we get negative, which we clip to 0.0.
            diff_voi = voi_doses - tolerance
            overdose_errors = np.maximum(diff_voi, 0.0)
            
            # Mean Squared Error of the overdosed voxels
            # This penalizes hot spots even if the mean is fine
            oar_penalty += np.mean(overdose_errors**2)

        # --- WEIGHTS ---
        # Since we are squaring errors, values might be smaller than before.
        # You might need to tune these.
        alpha = 5.0   # Weight for Target Underdose
        beta = 5.0    # Weight for OAR Overdose
        
        # Reward calculation
        # Max reward is 1.0 (Perfect plan).
        reward = 1.0 - (alpha * target_penalty + beta * oar_penalty)
        
        return float(np.clip(reward, -10.0, 10.0))

    @staticmethod
    def _create_target_mask(mask_volume, target_center, target_size):
        mask = mask_volume
        x0, y0, z0 = target_center
        dx, dy, dz = target_size
        x_start = max(0, x0 - dx // 2)
        x_end = min(mask_volume.shape[0], x0 + dx // 2 + 1)
        y_start = max(0, y0 - dy // 2)
        y_end = min(mask_volume.shape[1], y0 + dy // 2 + 1)
        z_start = max(0, z0 - dz // 2)
        z_end = min(mask_volume.shape[2], z0 + dz // 2 + 1)
        mask[x_start:x_end, y_start:y_end, z_start:z_end] = 1
        return np.isclose(mask, 1)

    @staticmethod
    def _create_voi_mask(mask_volume, voi_center, voi_size, weight):
        mask = mask_volume
        x0, y0, z0 = voi_center
        sx, sy, sz = voi_size
        x_start = max(0, x0 - sx // 2)
        x_end = min(mask_volume.shape[0], x0 + sx // 2 + 1)
        y_start = max(0, y0 - sy // 2)
        y_end = min(mask_volume.shape[1], y0 + sy // 2 + 1)
        z_start = max(0, z0 - sz // 2)
        z_end = min(mask_volume.shape[2], z0 + sz // 2 + 1)
        
        # Only overwrite empty space (0.0). Do not overwrite existing VOIs or Target.
        # Note: _create_target_mask is usually called last or logic needs care.
        # In reset(), we call _create_voi_mask FIRST, then target.
        # So here we just write. Target will overwrite later.
        
        region = mask[x_start:x_end, y_start:y_end, z_start:z_end]
        # Only write where it is currently 0 to prevent VOI-on-VOI overwrite issues
        # (though perturb_config tries to prevent physical overlap)
        region[region == 0] = float(weight)
        return mask

    def render(self, dose_threshold=0.05, beam_length=25, show=True, save_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-5, self.state[0].shape[0]+5)
        ax.set_ylim(-5, self.state[0].shape[1]+5)
        ax.set_zlim(-5, self.state[0].shape[2]+5)
        ax.set_box_aspect([1, 1, 1])
        cmap = cm.get_cmap("viridis")
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        # Base dose
        x, y, z = np.where(self.state[0] > 0)
        dose_vals = self.state[0][x, y, z]
        ax.scatter(x+0.5, y+0.5, z+0.5, c=cmap(norm(dose_vals)), alpha=0.1, s=10, zorder=1)

        # Target
        x_t, y_t, z_t = np.where(self.volume_mask == 1)
        if len(x_t) > 0:
            dose_vals_t = self.state[0][x_t, y_t, z_t]
            ax.scatter(x_t+0.5, y_t+0.5, z_t+0.5, c=cmap(norm(dose_vals_t)), marker='o', s=15, edgecolors='black', alpha=1.0, zorder=3)

        # VOIs
        for weight in self.weights:
            x_v, y_v, z_v = np.where(np.isclose(self.volume_mask, weight))
            if len(x_v) > 0:
                dose_vals_v = self.state[0][x_v, y_v, z_v]
                ax.scatter(x_v+0.5, y_v+0.5, z_v+0.5, c=cmap(norm(dose_vals_v)), marker='s', s=15, edgecolors='blue', alpha=1.0, zorder=4)

        # Beams
        t = np.linspace(0, beam_length, 20)
        for beam in self.beam_slots:
            if beam is None: continue
            src = np.array(beam["source_point"], dtype=float)
            ax.scatter(*src, color='red', marker='x', s=50, linewidths=2, zorder=5)
            for p0 in beam["raster_points"]:
                direction = np.array(beam["direction"], dtype=float)
                line = p0.reshape((3,1)) + np.outer(direction, t)
                ax.plot(line[0,:], line[1,:], line[2,:], color='cyan', alpha=0.4, linewidth=0.8, zorder=2)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        return (img * 255).astype(np.uint8)