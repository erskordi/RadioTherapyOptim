"""
Clinical Radiotherapy Evaluation Environment

Features:
1. Realistic anatomical scenarios (head & neck, prostate, lung)
2. OAR priority levels (serial vs parallel organs)
3. Clinical DVH constraints (D95, Dmax, V20, etc.)
4. Proper clinical metrics for evaluation
5. Clinical-style visualization (coverage + constraint violations)

OAR Types:
- Serial organs (spinal cord, brainstem): MAX dose matters - any hot spot is critical
- Parallel organs (lung, liver, kidney): MEAN dose and volume constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import gymnasium as gym
from gymnasium import spaces
from numba import njit, prange
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json


# -----------------------
# DATA STRUCTURES
# -----------------------
def perturb_scenario(scenario, volume_shape, max_shift=2, tolerance_noise=0.0, rng=None):
    """
    Randomizes a clinical scenario while preventing overlaps.
    
    Args:
        scenario: Dict with "target" and "oars" from scenario creator
        volume_shape: Tuple (nx, ny, nz)
        max_shift: Maximum voxel shift in any direction
        tolerance_noise: Std dev for perturbing dose tolerances (0 to disable)
        rng: numpy random generator
    
    Returns:
        New scenario dict with perturbed positions and tolerances
    """
    if rng is None:
        rng = np.random.default_rng()
    
    volume_shape = np.array(volume_shape)
    
    def clamp_center(center, size):
        """Clamp center so box stays within volume."""
        center = np.array(center, dtype=int)
        size = np.array(size, dtype=int)
        half = size // 2
        low = half
        high = volume_shape - (size - half)
        return tuple(np.clip(center, low, high - 1))
    
    def boxes_intersect(c1, s1, c2, s2, margin=0):
        """Check if two boxes intersect (with optional margin)."""
        for i in range(3):
            min1 = c1[i] - s1[i] // 2 - margin
            max1 = min1 + s1[i] + 2 * margin
            min2 = c2[i] - s2[i] // 2
            max2 = min2 + s2[i]
            if max1 <= min2 or max2 <= min1:
                return False
        return True
    
    def perturb_constraint(constraint, tolerance_noise, rng):
        """Create a copy of constraint with perturbed tolerances."""
        if tolerance_noise <= 0:
            return constraint
        
        # Create new constraint with same type
        if isinstance(constraint, TargetConstraint):
            return TargetConstraint(
                name=constraint.name,
                prescription_dose=constraint.prescription_dose,
                d95_min=constraint.d95_min,
                d5_max=constraint.d5_max,
                v100_min=constraint.v100_min
            )
        elif isinstance(constraint, OARConstraint):
            # Perturb dose limits slightly (within ±tolerance_noise relative)
            def perturb_val(val):
                if val is None:
                    return None
                noise = 1.0 + rng.normal(0, tolerance_noise)
                return float(np.clip(val * noise, 0.01, 10.0))
            
            new_v_dose = None
            if constraint.v_dose is not None:
                dose_thresh, vol_limit = constraint.v_dose
                new_v_dose = (perturb_val(dose_thresh), vol_limit)  # Keep volume % fixed
            
            return OARConstraint(
                name=constraint.name,
                priority=constraint.priority,
                organ_type=constraint.organ_type,
                max_dose=perturb_val(constraint.max_dose),
                mean_dose=perturb_val(constraint.mean_dose),
                d2_limit=perturb_val(constraint.d2_limit),
                v_dose=new_v_dose,
                color=constraint.color
            )
        return constraint
    
    # 1. Place Target first
    target_info = scenario["target"]
    target_center = np.array(target_info["center"], dtype=int)
    target_size = np.array(target_info["size"], dtype=int)
    
    shift = rng.integers(-max_shift, max_shift + 1, size=3)
    new_target_center = clamp_center(target_center + shift, target_size)
    
    new_target = {
        "center": new_target_center,
        "size": tuple(target_size),
        "constraint": perturb_constraint(target_info["constraint"], 0, rng)  # Don't perturb target
    }
    
    # 2. Place OARs with collision detection
    new_oars = []
    placed_boxes = [(new_target_center, target_size)]  # Start with target
    
    for oar_info in scenario["oars"]:
        oar_center = np.array(oar_info["center"], dtype=int)
        oar_size = np.array(oar_info["size"], dtype=int)
        
        # Try to find non-overlapping position
        best_center = None
        for attempt in range(15):
            shift = rng.integers(-max_shift, max_shift + 1, size=3)
            candidate = clamp_center(oar_center + shift, oar_size)
            
            # Check collisions with all placed structures
            collision = False
            for (placed_c, placed_s) in placed_boxes:
                if boxes_intersect(candidate, oar_size, placed_c, placed_s, margin=1):
                    collision = True
                    break
            
            if not collision:
                best_center = candidate
                break
        
        # Fallback: use original position if no valid spot found
        if best_center is None:
            best_center = clamp_center(oar_center, oar_size)
        
        placed_boxes.append((best_center, oar_size))
        
        new_oars.append({
            "center": best_center,
            "size": tuple(oar_size),
            "constraint": perturb_constraint(oar_info["constraint"], tolerance_noise, rng)
        })
    
    return {
        "target": new_target,
        "oars": new_oars,
        "name": scenario["name"]
    }
@dataclass
class OARConstraint:
    """Clinical constraint for an Organ at Risk."""
    name: str
    priority: str  # "critical", "high", "medium", "low"
    organ_type: str  # "serial" or "parallel"
    
    # Dose constraints (at least one should be set)
    max_dose: Optional[float] = None      # Dmax: No voxel should exceed this
    mean_dose: Optional[float] = None     # Dmean: Average dose limit
    d2_limit: Optional[float] = None      # D2%: Dose to hottest 2% of volume
    v_dose: Optional[Tuple[float, float]] = None  # (dose, volume%): e.g., V20 < 30%
    
    # Visualization
    color: str = "blue"
    
    def evaluate(self, doses: np.ndarray) -> Dict:
        """Evaluate all constraints for this OAR."""
        results = {
            "name": self.name,
            "priority": self.priority,
            "violations": [],
            "metrics": {},
            "passed": True
        }
        
        if len(doses) == 0:
            return results
        
        # Dmax constraint
        if self.max_dose is not None:
            actual_max = np.max(doses)
            results["metrics"]["Dmax"] = actual_max
            if actual_max > self.max_dose:
                results["violations"].append(f"Dmax: {actual_max:.2f} > {self.max_dose:.2f}")
                results["passed"] = False
        
        # Dmean constraint
        if self.mean_dose is not None:
            actual_mean = np.mean(doses)
            results["metrics"]["Dmean"] = actual_mean
            if actual_mean > self.mean_dose:
                results["violations"].append(f"Dmean: {actual_mean:.2f} > {self.mean_dose:.2f}")
                results["passed"] = False
        
        # D2% constraint (near-max dose)
        if self.d2_limit is not None:
            d2 = np.percentile(doses, 98)
            results["metrics"]["D2%"] = d2
            if d2 > self.d2_limit:
                results["violations"].append(f"D2%: {d2:.2f} > {self.d2_limit:.2f}")
                results["passed"] = False
        
        # Volume constraint (e.g., V20 < 30%)
        if self.v_dose is not None:
            dose_threshold, volume_limit = self.v_dose
            volume_fraction = np.mean(doses >= dose_threshold) * 100
            results["metrics"][f"V{int(dose_threshold*100)}"] = volume_fraction
            if volume_fraction > volume_limit:
                results["violations"].append(
                    f"V{int(dose_threshold*100)}: {volume_fraction:.1f}% > {volume_limit:.1f}%"
                )
                results["passed"] = False
        
        return results


@dataclass 
class TargetConstraint:
    """Clinical constraint for target (tumor)."""
    name: str
    prescription_dose: float = 1.0
    
    # Coverage requirements
    d95_min: float = 0.95      # D95 >= 95% of prescription
    d5_max: float = 1.07       # D5 <= 107% (hot spot limit)
    v100_min: float = 0.95     # V100 >= 95% (coverage)
    
    def evaluate(self, doses: np.ndarray) -> Dict:
        """Evaluate target coverage."""
        results = {
            "name": self.name,
            "metrics": {},
            "violations": [],
            "passed": True,
            "coverage": 0.0
        }
        
        if len(doses) == 0:
            results["passed"] = False
            return results
        
        rx = self.prescription_dose
        
        # D95: Dose covering 95% of target
        d95 = np.percentile(doses, 5)
        results["metrics"]["D95"] = d95
        if d95 < self.d95_min * rx:
            results["violations"].append(f"D95: {d95:.2f} < {self.d95_min * rx:.2f}")
            results["passed"] = False
        
        # D5: Dose to hottest 5% (hot spot)
        d5 = np.percentile(doses, 95)
        results["metrics"]["D5"] = d5
        if d5 > self.d5_max * rx:
            results["violations"].append(f"D5: {d5:.2f} > {self.d5_max * rx:.2f}")
            results["passed"] = False
        
        # V100: Volume receiving prescription dose
        v100 = np.mean(doses >= rx) * 100
        results["metrics"]["V100"] = v100
        results["coverage"] = v100
        if v100 < self.v100_min * 100:
            results["violations"].append(f"V100: {v100:.1f}% < {self.v100_min * 100:.1f}%")
            results["passed"] = False
        
        # Additional useful metrics
        results["metrics"]["Dmin"] = np.min(doses)
        results["metrics"]["Dmax"] = np.max(doses)
        results["metrics"]["Dmean"] = np.mean(doses)
        results["metrics"]["HI"] = (d5 - d95) / np.mean(doses)  # Homogeneity Index
        
        return results


# -----------------------
# CLINICAL SCENARIOS
# -----------------------

def create_head_neck_scenario(volume_shape=(18, 18, 18)):
    """
    Head & Neck cancer scenario.
    
    Challenging due to many critical OARs surrounding the target:
    - Spinal cord (serial, critical)
    - Brainstem (serial, critical)  
    - Parotid glands (parallel, for quality of life)
    - Optic structures (serial, critical)
    """
    center = np.array(volume_shape) // 2
    
    target = {
        "center": tuple(center),
        "size": (4, 4, 4),
        "constraint": TargetConstraint(
            name="Nasopharynx Tumor",
            prescription_dose=1.0,
            d95_min=0.95,
            v100_min=0.95
        )
    }
    
    oars = [
        {
            "center": (center[0], center[1] - 4, center[2]),
            "size": (2, 2, 10),
            "constraint": OARConstraint(
                name="Spinal Cord",
                priority="critical",
                organ_type="serial",
                max_dose=0.45,  # 45 Gy equivalent
                d2_limit=0.50,
                color="red"
            )
        },
        {
            "center": (center[0], center[1] + 3, center[2] + 2),
            "size": (3, 3, 3),
            "constraint": OARConstraint(
                name="Brainstem",
                priority="critical", 
                organ_type="serial",
                max_dose=0.54,
                color="darkred"
            )
        },
        {
            "center": (center[0] - 5, center[1], center[2]),
            "size": (2, 3, 3),
            "constraint": OARConstraint(
                name="Left Parotid",
                priority="high",
                organ_type="parallel",
                mean_dose=0.26,
                color="orange"
            )
        },
        {
            "center": (center[0] + 5, center[1], center[2]),
            "size": (2, 3, 3),
            "constraint": OARConstraint(
                name="Right Parotid",
                priority="high",
                organ_type="parallel",
                mean_dose=0.26,
                color="orange"
            )
        },
        {
            "center": (center[0], center[1] + 5, center[2] + 3),
            "size": (2, 2, 2),
            "constraint": OARConstraint(
                name="Optic Chiasm",
                priority="critical",
                organ_type="serial",
                max_dose=0.54,
                color="purple"
            )
        }
    ]
    
    return {"target": target, "oars": oars, "name": "Head & Neck"}


def create_prostate_scenario(volume_shape=(18, 18, 18)):
    """
    Prostate cancer scenario.
    
    OARs:
    - Rectum (serial/parallel, posterior)
    - Bladder (parallel, superior)
    - Femoral heads (parallel, lateral)
    """
    center = np.array(volume_shape) // 2
    
    target = {
        "center": tuple(center),
        "size": (4, 4, 3),
        "constraint": TargetConstraint(
            name="Prostate",
            prescription_dose=1.0,
            d95_min=0.95,
            v100_min=0.95
        )
    }
    
    oars = [
        {
            "center": (center[0], center[1] - 3, center[2]),
            "size": (3, 2, 5),
            "constraint": OARConstraint(
                name="Rectum",
                priority="high",
                organ_type="parallel",
                max_dose=0.75,
                mean_dose=0.50,
                v_dose=(0.70, 25.0),  # V70 < 25%
                color="brown"
            )
        },
        {
            "center": (center[0], center[1] + 4, center[2]),
            "size": (4, 3, 5),
            "constraint": OARConstraint(
                name="Bladder",
                priority="medium",
                organ_type="parallel",
                mean_dose=0.45,
                v_dose=(0.65, 50.0),  # V65 < 50%
                color="yellow"
            )
        },
        {
            "center": (center[0] - 5, center[1], center[2] - 2),
            "size": (3, 3, 3),
            "constraint": OARConstraint(
                name="Left Femoral Head",
                priority="low",
                organ_type="parallel",
                max_dose=0.50,
                color="cyan"
            )
        },
        {
            "center": (center[0] + 5, center[1], center[2] - 2),
            "size": (3, 3, 3),
            "constraint": OARConstraint(
                name="Right Femoral Head",
                priority="low",
                organ_type="parallel",
                max_dose=0.50,
                color="cyan"
            )
        }
    ]
    
    return {"target": target, "oars": oars, "name": "Prostate"}


def create_lung_scenario(volume_shape=(18, 18, 18)):
    """
    Lung cancer scenario (SBRT-style).
    
    Challenging due to:
    - Small target
    - Surrounding critical structures
    - Need for steep dose gradients
    """
    center = np.array(volume_shape) // 2
    # Offset target to one side (more realistic lung tumor position)
    target_center = (center[0] - 3, center[1], center[2])
    
    target = {
        "center": target_center,
        "size": (3, 3, 3),
        "constraint": TargetConstraint(
            name="Lung Tumor",
            prescription_dose=1.0,
            d95_min=0.95,
            d5_max=1.10,  # Allow slightly hotter for SBRT
            v100_min=0.95
        )
    }
    
    oars = [
        {
            "center": (center[0] + 2, center[1], center[2]),
            "size": (2, 2, 12),
            "constraint": OARConstraint(
                name="Spinal Cord",
                priority="critical",
                organ_type="serial",
                max_dose=0.20,  # Very strict for SBRT
                color="red"
            )
        },
        {
            "center": (center[0] + 3, center[1], center[2]),
            "size": (3, 4, 6),
            "constraint": OARConstraint(
                name="Heart",
                priority="high",
                organ_type="parallel",
                max_dose=0.30,
                mean_dose=0.10,
                color="pink"
            )
        },
        {
            "center": (center[0] + 1, center[1], center[2] + 3),
            "size": (2, 3, 3),
            "constraint": OARConstraint(
                name="Esophagus",
                priority="high",
                organ_type="serial",
                max_dose=0.35,
                color="green"
            )
        },
        {
            "center": (center[0] - 4, center[1], center[2]),
            "size": (6, 8, 10),
            "constraint": OARConstraint(
                name="Healthy Lung",
                priority="medium",
                organ_type="parallel",
                mean_dose=0.08,
                v_dose=(0.20, 30.0),  # V20 < 30%
                color="lightblue"
            )
        },
        {
            "center": (center[0], center[1] + 5, center[2]),
            "size": (3, 2, 4),
            "constraint": OARConstraint(
                name="Great Vessels",
                priority="high",
                organ_type="serial",
                max_dose=0.40,
                color="darkblue"
            )
        }
    ]
    
    return {"target": target, "oars": oars, "name": "Lung SBRT"}


# -----------------------
# NUMBA KERNELS
# -----------------------

@njit(fastmath=True)
def ray_box_intersection(p0, d, box_min, box_max):
    tmin, tmax = -1e30, 1e30
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if p0[i] < box_min[i] or p0[i] > box_max[i]:
                return -1.0, -1.0
        else:
            inv_d = 1.0 / d[i]
            t1 = (box_min[i] - p0[i]) * inv_d
            t2 = (box_max[i] - p0[i]) * inv_d
            if t1 < t2:
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
            else:
                tmin = max(tmin, t2)
                tmax = min(tmax, t1)
    if tmin > tmax or tmax < 0:
        return -1.0, -1.0
    return max(tmin, 0.0), tmax


@njit(fastmath=True)
def bragg_peak(depth, peak_depth):
    """Bragg peak with range straggling."""
    sigma_base = 0.3
    sigma = sigma_base * (1.0 + 0.05 * np.sqrt(max(peak_depth, 1.0)))
    
    entrance = 0.25
    diff = depth - peak_depth
    
    if depth < 0:
        return 0.0
    elif depth <= peak_depth:
        sigma_rise = peak_depth * 0.25
        rise = np.exp(-diff**2 / (2 * sigma_rise**2))
        return entrance + (1.0 - entrance) * rise
    else:
        fall = np.exp(-diff**2 / (2 * sigma**2))
        tail = 0.03 * np.exp(-diff / (sigma * 2))
        return fall * 0.97 + tail


@njit(parallel=True, fastmath=True)
def compute_dose(volume, raster_points, direction, intensities, energy_u, 
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
            t_entry, t_exit = ray_box_intersection(p0, direction, box_min, box_max)

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
                            bp_val = bragg_peak(depth_along, peak)
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
# BEAM CLASS
# -----------------------

def orthonormal_basis(v):
    v = v / (np.linalg.norm(v) + 1e-12)
    if abs(v[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    u = a - v * np.dot(a, v)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(v, u)
    return u, w


class Beam:
    def __init__(self, gantry, elevation, raster_grid, raster_spacing,
                 volume_shape, isocenter, setup_error=None):
        self.volume_shape = volume_shape
        self.isocenter = np.array(isocenter, dtype=float) + 0.5
        
        if setup_error is not None:
            self.isocenter += np.array(setup_error)
        
        cos_el = np.cos(elevation)
        direction = np.array([
            -cos_el * np.cos(gantry),
            -cos_el * np.sin(gantry),
            -np.sin(elevation)
        ])
        self.direction = direction / (np.linalg.norm(direction) + 1e-12)
        
        self.source_distance = np.linalg.norm(volume_shape) / 2 + 5
        self.source_point = self.isocenter - self.source_distance * self.direction
        
        # Create raster points
        box_min = np.zeros(3)
        box_max = np.array(volume_shape, dtype=float)
        t_entry, _ = ray_box_intersection(self.source_point, self.direction, box_min, box_max)
        
        if t_entry >= 0:
            plane_center = self.source_point + t_entry * self.direction
        else:
            plane_center = self.isocenter
        
        ny, nx = raster_grid
        sy, sx = raster_spacing
        u, w = orthonormal_basis(self.direction)
        
        y_off = (np.arange(ny) - (ny-1)/2) * sy
        x_off = (np.arange(nx) - (nx-1)/2) * sx
        xx, yy = np.meshgrid(x_off, y_off)
        
        pts = plane_center + xx[..., None] * u + yy[..., None] * w
        self.raster_points = pts.reshape(-1, 3)


# -----------------------
# CLINICAL ENVIRONMENT
# -----------------------

class ClinicalRadiotherapyEnv(gym.Env):
    """
    Radiotherapy environment with clinical constraints and evaluation.
    
    Features:
    - Multiple clinical scenarios (head & neck, prostate, lung)
    - Realistic OAR constraints (Dmax, Dmean, DVH)
    - Clinical metrics (D95, V100, HI)
    - Uncertainty modeling (range, setup)
    - Clinical-style visualization
    """
    
    SCENARIOS = {
        "head_neck": create_head_neck_scenario,
        "prostate": create_prostate_scenario,
        "lung": create_lung_scenario
    }
    
    def __init__(self, config):
        super().__init__()
        
        self.volume_shape = tuple(config.get("volume_shape", (18, 18, 18)))
        self.raster_grid = tuple(config.get("raster_grid", (4, 4)))
        self.raster_spacing = tuple(config.get("raster_spacing", (1.0, 1.0)))
        self.max_steps = config.get("max_steps", 3)
        self.num_layers = config.get("num_layers", 6)
        
        # Scenario selection
        self.scenario_name = config.get("scenario", "head_neck")
        self.randomize_scenario = config.get("randomize_scenario", False)
        
        # Uncertainty parameters
        self.range_uncertainty_std = config.get("range_uncertainty_std", 0.03)
        self.setup_uncertainty_std = config.get("setup_uncertainty_std", 0.3)
        self.apply_uncertainty = config.get("apply_uncertainty", True)
        
        # Reward weights
        self.coverage_weight = config.get("coverage_weight", 5.0)
        self.constraint_weight = config.get("constraint_weight", 2.0)
        self.critical_violation_penalty = config.get("critical_violation_penalty", 5.0)
        
        # Action space
        self.num_raster = self.raster_grid[0] * self.raster_grid[1]
        self.action_dim = 2 + self.num_layers + self.num_layers * self.num_raster
        
        action_low = np.concatenate([
            np.array([-np.pi, -np.pi/2]),
            np.zeros(self.num_layers),
            np.zeros(self.num_layers * self.num_raster)
        ]).astype(np.float32)
        
        action_high = np.concatenate([
            np.array([np.pi, np.pi/2]),
            np.ones(self.num_layers),
            np.ones(self.num_layers * self.num_raster)
        ]).astype(np.float32)
        
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # Observation space
        self.observation_space = spaces.Dict({
            "dose": spaces.Box(0, np.inf, shape=(2,) + self.volume_shape, dtype=np.float32),
            "step": spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        })
        
        # State
        self.scenario = None
        self.target_constraint = None
        self.oar_constraints = []
        self.structure_masks = {}
        self.dose_total = None
        self.beam_slots = []
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Select scenario
        if self.randomize_scenario:
            scenario_name = self.np_random.choice(list(self.SCENARIOS.keys()))
        else:
            scenario_name = self.scenario_name
        
        base_scenario = self.SCENARIOS[scenario_name](self.volume_shape)
        self.scenario = perturb_scenario(
            base_scenario,
            self.volume_shape,
            max_shift=1,
            tolerance_noise=0.0,
            rng=self.np_random
            )
        # Setup target
        target_info = self.scenario["target"]
        self.target_center = np.array(target_info["center"])
        self.target_size = target_info["size"]
        self.target_constraint = target_info["constraint"]
        
        # Setup OARs
        self.oar_constraints = []
        self.oar_infos = []
        for oar_info in self.scenario["oars"]:
            self.oar_constraints.append(oar_info["constraint"])
            self.oar_infos.append(oar_info)
        
        
        # Create masks
        self._create_masks()
        
        # Initialize dose
        self.dose_total = np.zeros(self.volume_shape, dtype=np.float32)
        self.beam_slots = []
        self.step_count = 0
        
        # Initial costs
        self.prev_target_cost = self._get_target_underdose()
        self.prev_violation_cost = 0.0
        
        state = np.stack([self.dose_total, self.structure_masks["combined"]], axis=0)
        return {"dose": state.astype(np.float32), "step": np.array([0.0], dtype=np.float32)}, {}

    def _create_masks(self):
        """Create structure masks."""
        self.structure_masks = {}
        combined = np.zeros(self.volume_shape, dtype=np.float32)
        
        # Target mask (value = 1.0)
        target_mask = np.zeros(self.volume_shape, dtype=bool)
        self._fill_box(target_mask, self.target_center, self.target_size, True)
        self.structure_masks["target"] = target_mask
        combined[target_mask] = 1.0
        
        # OAR masks (values = 0.1 to 0.9 based on priority)
        priority_values = {"critical": 0.2, "high": 0.4, "medium": 0.6, "low": 0.8}
        
        for constraint, info in zip(self.oar_constraints, self.oar_infos):
            oar_mask = np.zeros(self.volume_shape, dtype=bool)
            center = np.array(info["center"])
            # Small perturbation
            #center = np.clip(center + self.np_random.integers(-1, 2, size=3), 0, 
            #               np.array(self.volume_shape) - 1)
            self._fill_box(oar_mask, center, info["size"], True)
            
            # Don't overlap with target
            oar_mask = oar_mask & ~target_mask
            
            self.structure_masks[constraint.name] = oar_mask
            combined[oar_mask] = priority_values.get(constraint.priority, 0.5)
        
        self.structure_masks["combined"] = combined

    def _fill_box(self, mask, center, size, value):
        """Fill a box region in mask."""
        c = np.array(center)
        s = np.array(size)
        slices = tuple(slice(max(0, int(c[i] - s[i]//2)), 
                            min(mask.shape[i], int(c[i] + s[i]//2 + 1))) 
                      for i in range(3))
        mask[slices] = value

    def step(self, action):
        action = np.asarray(action).ravel()
        
        gantry = float(action[0])
        elevation = float(action[1])
        energies = np.clip(action[2:2+self.num_layers], 0, 1)
        intensities = np.clip(action[2+self.num_layers:], 0, 1)
        
        # Sample uncertainties
        if self.apply_uncertainty:
            range_factor = 1.0 + self.np_random.normal(0, self.range_uncertainty_std)
            setup_error = self.np_random.normal(0, self.setup_uncertainty_std, size=3)
        else:
            range_factor = 1.0
            setup_error = None
        
        # Create and apply beam
        beam = Beam(gantry, elevation, self.raster_grid, self.raster_spacing,
                   self.volume_shape, self.target_center, setup_error)
        
        dose_contrib = np.zeros(self.volume_shape, dtype=np.float32)
        compute_dose(
            dose_contrib,
            np.ascontiguousarray(beam.raster_points),
            np.ascontiguousarray(beam.direction),
            np.ascontiguousarray(intensities),
            np.ascontiguousarray(energies),
            np.array(self.volume_shape),
            self.num_layers,
            range_factor
        )
        
        self.dose_total += dose_contrib
        self.beam_slots.append({
            "gantry": gantry, "elevation": elevation,
            "direction": beam.direction, "source_point": beam.source_point,
            "raster_points": beam.raster_points
        })
        
        # Compute reward
        reward = self._compute_reward()
        
        self.step_count += 1
        eval_results = self.evaluate()
        summary = eval_results["summary"]
    
        # Check termination conditions
        target_achieved = summary["target_passed"]
        no_critical_violations = summary["critical_violations"] == 0
        plan_acceptable = summary["plan_acceptable"]  # target_passed AND no critical violations
    
        max_steps_reached = self.step_count >= self.max_steps
    
        # Early termination if plan is acceptable
        done = max_steps_reached or target_achieved
        #done = self.step_count >= self.max_steps
        
        if done:
            if plan_acceptable:
                terminal_reward = self._compute_terminal_reward(eval_results)
                reward += terminal_reward
            else:
                penalty_violation = -1
                reward+=penalty_violation
        
        state = np.stack([self.dose_total, self.structure_masks["combined"]], axis=0)
        return {"dose": state.astype(np.float32), 
                "step": np.array([self.step_count / self.max_steps], dtype=np.float32)}, \
               float(reward), done, False, {}

    def _compute_reward(self):
        """Step reward based on improvement."""
        # Target improvement
        curr_underdose = self._get_target_underdose()
        target_improvement = self.prev_target_cost - curr_underdose
        self.prev_target_cost = curr_underdose
        
        # Constraint violation increase
        curr_violation = self._get_total_violation_cost()
        violation_increase = curr_violation - self.prev_violation_cost
        self.prev_violation_cost = curr_violation
        
        reward = self.coverage_weight * target_improvement - self.constraint_weight * violation_increase
        return reward * 10.0  # Scale

    def _compute_terminal_reward(self, eval_results):
        """Terminal reward based on final plan quality."""
        reward = 0.0
        
        # Coverage bonus
        coverage = eval_results["target"]["coverage"]
        reward += self.coverage_weight * coverage / 100.0
        
        # Constraint violation penalties
        for oar_name, oar_result in eval_results["oars"].items():
            if not oar_result["passed"]:
                if oar_result["priority"] == "critical":
                    reward -= self.critical_violation_penalty
                elif oar_result["priority"] == "high":
                    reward -= self.constraint_weight
                else:
                    reward -= self.constraint_weight * 0.5
        
        return reward

    def _get_target_underdose(self):
        """Get mean underdose in target."""
        target_doses = self.dose_total[self.structure_masks["target"]]
        if len(target_doses) == 0:
            return 1.0
        underdose = np.maximum(self.target_constraint.prescription_dose - target_doses, 0)
        return np.mean(underdose)

    def _get_total_violation_cost(self):
        """Sum of constraint violation magnitudes."""
        cost = 0.0
        for constraint in self.oar_constraints:
            mask = self.structure_masks.get(constraint.name)
            if mask is None or not mask.any():
                continue
            doses = self.dose_total[mask]
            
            if constraint.max_dose is not None:
                excess = np.maximum(np.max(doses) - constraint.max_dose, 0)
                cost += excess
            
            if constraint.mean_dose is not None:
                excess = np.maximum(np.mean(doses) - constraint.mean_dose, 0)
                cost += excess
        
        return cost

    def evaluate(self) -> Dict:
        """Full clinical evaluation of current plan."""
        results = {
            "scenario": self.scenario["name"],
            "target": {},
            "oars": {},
            "summary": {}
        }
        
        # Evaluate target
        target_doses = self.dose_total[self.structure_masks["target"]]
        results["target"] = self.target_constraint.evaluate(target_doses)
        
        # Evaluate OARs
        n_violations = 0
        critical_violations = 0
        
        for constraint in self.oar_constraints:
            mask = self.structure_masks.get(constraint.name)
            if mask is None or not mask.any():
                continue
            
            oar_doses = self.dose_total[mask]
            oar_result = constraint.evaluate(oar_doses)
            results["oars"][constraint.name] = oar_result
            
            if not oar_result["passed"]:
                n_violations += 1
                if constraint.priority == "critical":
                    critical_violations += 1
        
        # Summary
        results["summary"] = {
            "target_coverage": results["target"]["coverage"],
            "target_passed": results["target"]["passed"],
            "n_oar_violations": n_violations,
            "critical_violations": critical_violations,
            "plan_acceptable": results["target"]["passed"] and critical_violations == 0
        }
        
        return results

    def render(self, mode="clinical"):
        """
        Clinical-style rendering.
        
        Colors:
        - Target: Green (covered) / Black (underdosed)
        - OARs: Green (within constraint) / Red (violated)
        - Dose cloud: Viridis colormap
        """
        fig = plt.figure(figsize=(16, 6))
        
        # Get evaluation
        eval_results = self.evaluate()
        rx = self.target_constraint.prescription_dose
        
        # 1. 3D View
        ax1 = fig.add_subplot(131, projection='3d')
        self._render_3d(ax1, eval_results, rx)
        
        # 2. Axial slice
        ax2 = fig.add_subplot(132)
        self._render_slice(ax2, eval_results, rx, axis=2)
        ax2.set_title('Axial Slice (z=center)')
        
        # 3. DVH
        ax3 = fig.add_subplot(133)
        self._render_dvh(ax3)
        
        # Title with summary
        summary = eval_results["summary"]
        title = f"{self.scenario['name']} | "
        title += f"Coverage: {summary['target_coverage']:.1f}% | "
        title += f"Violations: {summary['n_oar_violations']} "
        title += f"(Critical: {summary['critical_violations']})"
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img = plt.imread(buf)
        plt.close(fig)
        return (img * 255).astype(np.uint8)

    def _render_3d(self, ax, eval_results, rx):
        """Render 3D view with clinical coloring."""
        ax.set_xlim(0, self.volume_shape[0])
        ax.set_ylim(0, self.volume_shape[1])
        ax.set_zlim(0, self.volume_shape[2])
        
        # Target voxels
        target_mask = self.structure_masks["target"]
        x, y, z = np.where(target_mask)
        if len(x) > 0:
            doses = self.dose_total[target_mask]
            colors = np.where(doses >= rx * 0.95, 'green', 'black')
            ax.scatter(x+0.5, y+0.5, z+0.5, c=colors, s=50, alpha=0.8, 
                      edgecolors='darkgreen', linewidths=0.5)
        
        # OAR voxels
        for constraint in self.oar_constraints:
            mask = self.structure_masks.get(constraint.name)
            if mask is None or not mask.any():
                continue
            
            oar_result = eval_results["oars"].get(constraint.name, {})
            passed = oar_result.get("passed", True)
            
            x, y, z = np.where(mask)
            if len(x) > 0:
                doses = self.dose_total[mask]
                
                # Color based on violation
                if constraint.max_dose is not None:
                    violated = doses > constraint.max_dose
                elif constraint.mean_dose is not None:
                    violated = np.full_like(doses, not passed, dtype=bool)
                else:
                    violated = np.zeros_like(doses, dtype=bool)
                
                colors = np.where(violated, 'red', 'lightgreen')
                ax.scatter(x+0.5, y+0.5, z+0.5, c=colors, s=20, alpha=0.5,
                          marker='s', edgecolors='gray', linewidths=0.3)
        
        # Beams
        for i, beam in enumerate(self.beam_slots):
            src = beam["source_point"]
            ax.scatter(*src, c='blue', marker='x', s=100)
            
            for p0 in beam["raster_points"][::4]:
                t = np.linspace(0, 20, 10)
                line = p0.reshape(3,1) + np.outer(beam["direction"], t)
                ax.plot(line[0], line[1], line[2], 'c-', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D View')

    def _render_slice(self, ax, eval_results, rx, axis=2):
        """Render 2D slice with clinical coloring."""
        mid = self.volume_shape[axis] // 2
        
        if axis == 2:
            dose_slice = self.dose_total[:, :, mid]
            combined_slice = self.structure_masks["combined"][:, :, mid]
        elif axis == 1:
            dose_slice = self.dose_total[:, mid, :]
            combined_slice = self.structure_masks["combined"][:, mid, :]
        else:
            dose_slice = self.dose_total[mid, :, :]
            combined_slice = self.structure_masks["combined"][mid, :, :]
        
        # Create RGB image
        rgb = np.zeros(dose_slice.shape + (3,))
        
        # Dose colormap (background)
        dose_norm = np.clip(dose_slice / (rx * 1.2), 0, 1)
        rgb[..., 0] = dose_norm  # Red channel for dose
        rgb[..., 1] = dose_norm * 0.5
        rgb[..., 2] = 0
        
        # Overlay structures
        # Target: green if covered, black if cold
        target_slice = self.structure_masks["target"]
        if axis == 2:
            target_slice = target_slice[:, :, mid]
        elif axis == 1:
            target_slice = target_slice[:, mid, :]
        else:
            target_slice = target_slice[mid, :, :]
        
        covered = (dose_slice >= rx * 0.95) & target_slice
        cold = (dose_slice < rx * 0.95) & target_slice
        
        rgb[covered] = [0, 0.8, 0]  # Green
        rgb[cold] = [0.1, 0.1, 0.1]  # Dark gray/black
        
        # OARs: green if OK, red if violated
        for constraint in self.oar_constraints:
            mask = self.structure_masks.get(constraint.name)
            if mask is None:
                continue
            
            if axis == 2:
                oar_slice = mask[:, :, mid]
            elif axis == 1:
                oar_slice = mask[:, mid, :]
            else:
                oar_slice = mask[mid, :, :]
            
            oar_result = eval_results["oars"].get(constraint.name, {})
            
            if constraint.max_dose is not None:
                violated = (dose_slice > constraint.max_dose) & oar_slice
                ok = (dose_slice <= constraint.max_dose) & oar_slice
            else:
                ok = oar_slice
                violated = np.zeros_like(oar_slice)
            
            rgb[ok] = [0.5, 1.0, 0.5]  # Light green
            rgb[violated] = [1.0, 0.2, 0.2]  # Red
        
        ax.imshow(rgb.transpose(1, 0, 2), origin='lower')
        
        # Add contours
        ax.contour(target_slice.T, levels=[0.5], colors='green', linewidths=2)
        for constraint in self.oar_constraints:
            mask = self.structure_masks.get(constraint.name)
            if mask is None:
                continue
            if axis == 2:
                oar_slice = mask[:, :, mid]
            elif axis == 1:
                oar_slice = mask[:, mid, :]
            else:
                oar_slice = mask[mid, :, :]
            ax.contour(oar_slice.T, levels=[0.5], colors='blue', linewidths=1, linestyles='--')

    def _render_dvh(self, ax):
        """Render Dose-Volume Histogram."""
        # Target DVH
        target_doses = self.dose_total[self.structure_masks["target"]]
        if len(target_doses) > 0:
            sorted_doses = np.sort(target_doses)[::-1]
            volumes = np.linspace(0, 100, len(sorted_doses))
            ax.plot(sorted_doses, volumes, 'g-', linewidth=2, label='Target')
        
        # OAR DVHs
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.oar_constraints)))
        for i, constraint in enumerate(self.oar_constraints):
            mask = self.structure_masks.get(constraint.name)
            if mask is None or not mask.any():
                continue
            
            oar_doses = self.dose_total[mask]
            sorted_doses = np.sort(oar_doses)[::-1]
            volumes = np.linspace(0, 100, len(sorted_doses))
            
            style = '-' if constraint.priority in ['critical', 'high'] else '--'
            ax.plot(sorted_doses, volumes, style, color=colors[i], 
                   linewidth=1.5, label=constraint.name[:15])
            
            # Mark Dmax constraint
            if constraint.max_dose is not None:
                ax.axvline(constraint.max_dose, color=colors[i], linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Dose')
        ax.set_ylabel('Volume (%)')
        ax.set_title('DVH')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 100)

    def print_evaluation(self):
        """Print detailed evaluation report."""
        results = self.evaluate()
        
        print("\n" + "="*60)
        print(f"TREATMENT PLAN EVALUATION: {results['scenario']}")
        print("="*60)
        
        # Target
        t = results["target"]
        print(f"\n📎 TARGET: {t['name']}")
        print(f"   Coverage (V100): {t['coverage']:.1f}%")
        for metric, value in t["metrics"].items():
            print(f"   {metric}: {value:.3f}")
        if t["violations"]:
            print(f"   ⚠️  Violations: {', '.join(t['violations'])}")
        else:
            print(f"   ✅ All constraints met")
        
        # OARs
        print(f"\n📋 ORGANS AT RISK:")
        for name, oar in results["oars"].items():
            status = "✅" if oar["passed"] else "❌"
            print(f"\n   {status} {name} [{oar['priority']}]")
            for metric, value in oar["metrics"].items():
                print(f"      {metric}: {value:.3f}")
            if oar["violations"]:
                for v in oar["violations"]:
                    print(f"      ⚠️  {v}")
        
        # Summary
        s = results["summary"]
        print(f"\n" + "-"*40)
        print(f"SUMMARY:")
        print(f"   Target Coverage: {s['target_coverage']:.1f}%")
        print(f"   OAR Violations: {s['n_oar_violations']}")
        print(f"   Critical Violations: {s['critical_violations']}")
        print(f"   Plan Acceptable: {'✅ YES' if s['plan_acceptable'] else '❌ NO'}")
        print("="*60 + "\n")
        
        return results
