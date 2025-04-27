import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

# Set up logging
logger = logging.getLogger("G1Env")

# Get base directory for consistent file references
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class G1EnvConfig:
    """Configuration for G1 environment parameters"""
    # Goal parameters
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.7]))
    # goal_pos: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.0, 0.0]))
    goal_success_threshold: float = 0.25
    goal_reward_weight: float = 50.0
    
    # Stability parameters
    stability_reward_weight: float = 4.0
    nominal_height: float = 0.7
    fall_threshold: float = 0.2
    early_termination_penalty: float = 10.0
    
    # Contact reward
    contact_reward_weight: float = 1.0
    
    # Penalty weights
    torque_penalty_weight: float = 0.001
    joint_limit_penalty_weight: float = 0.05
    forward_lean_penalty_weight: float = 2.0
    velocity_penalty_weight: float = 0.01
    
    # Tracking weights
    lin_vel_tracking_weight: float = 1.0
    ang_vel_tracking_weight: float = 1.0
    height_penalty_weight: float = 1.5
    pose_similarity_weight: float = 0.5
    action_rate_penalty_weight: float = 0.01
    vertical_vel_penalty_weight: float = 1.0
    roll_pitch_penalty_weight: float = 1.0
    orientation_penalty_weight: float = 0.0  # absolute tilt penalty weight
    
    # Environment parameters
    frame_skip: int = 5
    action_scale: float = 0.015
    episode_length: int = 1000
    damping_factor: float = 0.98

class G1Env(gym.Env):
    """
    Custom Gym environment for the Unitree G1 robot
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, config=None):
        # Initialize configuration
        self.config = config if config is not None else G1EnvConfig()
        
        # Env parameters from config
        self.frame_skip = self.config.frame_skip
        self.goal_pos = self.config.goal_pos
        self.goal_success_threshold = self.config.goal_success_threshold
        self.goal_reward_weight = self.config.goal_reward_weight
        self.stability_reward_weight = self.config.stability_reward_weight
        self.nominal_height = self.config.nominal_height
        self.fall_threshold = self.config.fall_threshold
        self.early_termination_penalty = self.config.early_termination_penalty
        self.contact_reward_weight = self.config.contact_reward_weight
        self.torque_penalty_weight = self.config.torque_penalty_weight
        self.joint_limit_penalty_weight = self.config.joint_limit_penalty_weight
        self.forward_lean_penalty_weight = self.config.forward_lean_penalty_weight
        self.velocity_penalty_weight = self.config.velocity_penalty_weight
        self.lin_vel_tracking_weight = self.config.lin_vel_tracking_weight
        self.ang_vel_tracking_weight = self.config.ang_vel_tracking_weight
        self.height_penalty_weight = self.config.height_penalty_weight
        self.pose_similarity_weight = self.config.pose_similarity_weight
        self.action_rate_penalty_weight = self.config.action_rate_penalty_weight
        self.vertical_vel_penalty_weight = self.config.vertical_vel_penalty_weight
        self.roll_pitch_penalty_weight = self.config.roll_pitch_penalty_weight
        self.orientation_penalty_weight = self.config.orientation_penalty_weight
        self.action_scale = self.config.action_scale
        self.episode_length = self.config.episode_length
        self.damping_factor = self.config.damping_factor
        
        # Default commanded velocities and height (can be updated)
        self.vx_ref = 0.0  # Default target x velocity
        self.vy_ref = 0.0  # Default target y velocity
        self.wz_ref = 0.0  # Default target yaw velocity
        self.z_ref = self.nominal_height  # Default target height
        
        # Store previous actions for action rate penalty
        self.prev_action = None
        
        # Default joint positions (standing pose)
        self.default_joint_pos = None  # Will be set in reset
        
        # Tracking variables
        self.steps = 0
        self.render_mode = render_mode
        
        # Load the model
        model_path = os.path.join(BASE_DIR, "data/g1_robot/g1_23dof_simplified.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set joint damping for more stability (prevents wild oscillations)
        for i in range(self.model.njnt):
            # Skip the first 7 DoFs which are the floating base
            if i >= 7:
                # Damping coefficient for joint - start with 1.0 (moderate damping)
                self.model.dof_damping[i] = 1.0
        
        # Increase friction parameters
        for i in range(self.model.ngeom):
            self.model.geom_friction[i, 0] = 2.0  # Sliding friction (higher = more friction)
            self.model.geom_friction[i, 1] = 0.1  # Torsional friction
            self.model.geom_friction[i, 2] = 0.1  # Rolling friction
        
        # Get the actual dimensions from the model
        self.qpos_dim = self.model.nq
        self.qvel_dim = self.model.nv
        self.obs_dim = self.qpos_dim + self.qvel_dim
        # Assuming first 7 DOFs are free-floating and the rest are actuated joints
        self.act_dim = self.qvel_dim - 7
        
        logger.info(f"Loaded model with qpos dimension: {self.qpos_dim}, qvel dimension: {self.qvel_dim}")
        
        # For compatibility with training scripts, we need to maintain fixed dimensions
        # Use fixed dimensions that match expert buffer
        use_fixed_dims = True
        
        if use_fixed_dims:
            # Fixed dimensions for compatibility with expert buffer
            obs_dim = 60  # Fixed at 60 to match expert buffer
            act_dim = 30  # Fixed at 30 to match expert buffer
            logger.info(f"Using fixed observation dimension: {obs_dim}, action dimension: {act_dim} (for compatibility)")
        else:
            # Dynamic dimensions based on model properties
            obs_dim = self.obs_dim
            act_dim = self.act_dim
            logger.info(f"Using dynamic observation dimension: {obs_dim}, action dimension: {act_dim}")
        
        # Define action space (scaled to [-1, 1] by convention)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        
        # Define observation space
        high = np.ones(obs_dim, dtype=np.float32) * np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32
        )
        
        # Set up rendering
        self.viewer = None
        
        # Initialize renderer if human mode is requested
        if self.render_mode == "human":
            self._setup_renderer()
            
        # For seeding
        self.np_random = np.random.RandomState()
        self.initial_height = None
        
        # Get foot geom IDs for contact detection
        self.foot_geom_ids = set()
        self.foot_site_ids = set()
        
        # Find foot contact geoms by name first
        foot_keywords = ['foot', 'ankle', 'toe']
        
        if hasattr(self.model, 'geom_names'):
            for i, name in enumerate(self.model.geom_names):
                name_str = str(name).lower()
                if any(keyword in name_str for keyword in foot_keywords):
                    self.foot_geom_ids.add(i)
                    
        # If no foot geoms found by name, fall back to heuristics
        if not self.foot_geom_ids:
            for i in range(self.model.ngeom):
                # Use heuristics (spheres with small size near the ground)
                if self.model.geom_type[i] == 2:  # Type 2 is sphere in MuJoCo
                    if self.model.geom_size[i, 0] < 0.05:  # Small spheres
                        self.foot_geom_ids.add(i)
        
        # Use foot site IDs if available
        if hasattr(self.model, 'site_names'):
            for i, name in enumerate(self.model.site_names):
                name_str = str(name).lower()
                if any(keyword in name_str for keyword in foot_keywords):
                    self.foot_site_ids.add(i)
        
        logger.info(f"Found {len(self.foot_geom_ids)} foot contact geoms and {len(self.foot_site_ids)} foot sites")
    
    def seed(self, seed=None):
        """Set random seed"""
        self.np_random.seed(seed)
        # Reset MuJoCo's internal RNG if available
        try:
            mujoco.mj_resetSeed(self.model, self.data, seed)
        except AttributeError:
            # Try alternate seeding methods if available
            if hasattr(mujoco, 'mj_resetData'):
                # Reset data and then set the seed
                mujoco.mj_resetData(self.model, self.data)
        
        return [seed]
    
    def _setup_renderer(self):
        """Set up the MuJoCo renderer"""
        import glfw
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        # Create window
        self.window = glfw.create_window(1200, 900, "G1 Robot", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        # Make context current
        glfw.make_context_current(self.window)
        # Create renderer with default offscreen framebuffer size
        self.renderer = mujoco.Renderer(self.model)
    
    def _get_obs(self):
        """Get the current observation"""
        # Get joint positions (qpos) and velocities (qvel)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Clip velocities to prevent extreme values
        qvel = np.clip(qvel, -10.0, 10.0)
        
        # Concatenate to get the full state
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        
        # Make sure observation matches the expected dimension
        expected_dim = self.observation_space.shape[0]
        actual_dim = len(obs)
        
        if actual_dim < expected_dim:
            # If actual dimension is less than expected, pad with zeros
            padding = np.zeros(expected_dim - actual_dim, dtype=np.float32)
            obs = np.concatenate([obs, padding])
            if self.steps == 0:  # Only log once during initialization
                logger.debug(f"Padded observation from {actual_dim} to {expected_dim} dimensions")
        elif actual_dim > expected_dim:
            # If actual dimension is more than expected, truncate
            obs = obs[:expected_dim]
            if self.steps == 0:  # Only log once during initialization
                logger.debug(f"Truncated observation from {actual_dim} to {expected_dim} dimensions")
        
        # Replace any NaN or Inf values with zeros
        if np.any(~np.isfinite(obs)):
            nan_count = np.sum(np.isnan(obs))
            inf_count = np.sum(np.isinf(obs))
            if nan_count > 0 or inf_count > 0:
                logger.debug(f"Found {nan_count} NaN and {inf_count} Inf values in observation")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
        
    def reset(self, seed=None, options=None):
        """Reset the environment to a random initial state"""
        if seed is not None:
            self.seed(seed)
        
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Increase friction to prevent slipping
        for i in range(self.model.ngeom):
            if self.model.geom_solref is not None and i < len(self.model.geom_solref):
                # Increase friction coefficient
                self.model.geom_friction[i, 0] = 2.0  # Sliding friction 
                self.model.geom_friction[i, 1] = 0.1  # Torsional friction
                self.model.geom_friction[i, 2] = 0.1  # Rolling friction
                
                # Add more damping
                if hasattr(self.model, 'geom_solref') and i < len(self.model.geom_solref):
                    self.model.geom_solref[i, 0] = 0.04  # Add more damping
        
        # Reset step counter
        self.steps = 0
        
        # Neutralize only joint DOFs (preserve first 7 qpos for root)
        n_root = 7  # 3 for root pos, 4 for root quaternion
        self.data.qpos[n_root:] = 0.0
        self.data.qvel[:] = 0.0  # Reset all velocities to zero
        
        # Add slight random noise to joints for exploration
        joint_noise = self.np_random.uniform(-0.01, 0.01, size=self.qpos_dim-n_root)
        self.data.qpos[n_root:] += joint_noise
        
        # Save initial joint positions as default
        joint_start_idx = min(n_root, self.qpos_dim - self.act_dim)  # More robust indexing
        self.default_joint_pos = self.data.qpos[joint_start_idx:joint_start_idx+self.act_dim].copy()
        
        # Reset previous action
        self.prev_action = np.zeros(self.act_dim)
        
        # Forward dynamics to get the simulation into a valid state
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial height for comparison
        self.initial_height = self.data.qpos[2]
        
        # Get observation
        obs = self._get_obs()
        
        # Return observation and empty info dict (Gymnasium API)
        return obs, {}
    
    def _check_foot_contacts(self):
        """Check if any foot is in contact with the ground using precomputed geom IDs"""
        # Fast contact checking using precomputed foot geom IDs
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in self.foot_geom_ids or contact.geom2 in self.foot_geom_ids:
                return True
        
        # Alternative check using foot sites - check if they're close to the ground
        if not self.foot_site_ids:
            return False
            
        for site_id in self.foot_site_ids:
            site_pos = self.data.site_xpos[site_id]
            if site_pos[2] < 0.1:  # Site is close to ground
                return True
                
        return False
    
    def step(self, action):
        """Step the simulation forward based on the action"""
        self.steps += 1
        
        # Ensure action has the right dimension (matches action space)
        action_space_dim = self.action_space.shape[0]  # Should be 30 for compatibility
        actual_joints_dim = self.act_dim  # Actual number of actuated joints (22)
        
        if len(action) != action_space_dim:
            # If action is too short, pad with zeros
            if len(action) < action_space_dim:
                action = np.concatenate([action, np.zeros(action_space_dim - len(action), dtype=action.dtype)])
            # If action is too long, truncate
            else:
                action = action[:action_space_dim]
        
        # Scale and clip action for stability
        action = np.clip(action, -1.0, 1.0) * self.action_scale
        
        # Record the raw action for torque penalty calculation
        raw_action = action.copy()
        
        # Calculate action rate penalty if we have a previous action
        action_rate_penalty = 0.0
        if self.prev_action is not None:
            # Use actual joint dimensions for action rate penalty
            action_diff = raw_action[:actual_joints_dim] - self.prev_action
            action_rate_penalty = -self.action_rate_penalty_weight * np.sum(np.square(action_diff))
        
        # Update previous action for next step (store only the actual joint dimensions)
        self.prev_action = raw_action[:actual_joints_dim].copy()
        
        # We can only control the actual joint DOFs, not the root position/orientation
        # Skip the first 7 qpos values (3 for position, 4 for quaternion orientation)
        joint_start_idx = min(7, self.qpos_dim - actual_joints_dim)
        
        # Record root position before simulation for reward calculation
        prev_root_pos = self.data.qpos[:3].copy()
        prev_joint_pos = self.data.qpos[joint_start_idx:joint_start_idx+actual_joints_dim].copy()
        
        # Apply actions to the joints - only use the first actual_joints_dim values
        # This ensures we only use the number of actions the model can actually handle
        for i in range(actual_joints_dim):
            if i < len(action):  # Safety check
                self.data.qpos[joint_start_idx + i] += action[i]
        
        # Add joint limit constraint to prevent extreme values
        joint_limits = 3.14  # approx pi, reasonable joint limit in radians
        joint_pos = self.data.qpos[joint_start_idx:joint_start_idx+actual_joints_dim]
        
        # Store values for joint limit penalty calculation
        upper_limits = np.ones_like(joint_pos) * joint_limits
        lower_limits = -upper_limits
        
        # Calculate joint limit violations for penalty
        joint_limit_violations = np.sum(
            np.maximum(0, joint_pos - upper_limits)**2 + 
            np.maximum(0, lower_limits - joint_pos)**2
        )
        
        # Apply clipping to enforce joint limits
        self.data.qpos[joint_start_idx:joint_start_idx+actual_joints_dim] = np.clip(
            self.data.qpos[joint_start_idx:joint_start_idx+actual_joints_dim],
            -joint_limits, joint_limits
        )

        # Run the simulation with standard mj_step
        try:
            for _ in range(self.frame_skip):
                # Check if state is valid before stepping
                if np.any(~np.isfinite(self.data.qpos)) or np.any(~np.isfinite(self.data.qvel)):
                    logger.debug("Invalid state detected (NaN/Inf in state)")
                    # Reset to previous valid state
                    self.data.qpos[:] = np.nan_to_num(self.data.qpos, nan=0.0, posinf=0.0, neginf=0.0)
                    self.data.qvel[:] = np.nan_to_num(self.data.qvel, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Step physics with proper damping
                mujoco.mj_step(self.model, self.data)
                
                # Add damping to velocities for stability
                self.data.qvel[:] *= self.damping_factor
                
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            # Return early with terminated=True if simulation fails
            return self._get_obs(), -self.early_termination_penalty, True, False, {"error": str(e)}

        # Get the new observation
        obs = self._get_obs()

        # Get state information
        root_pos = self.data.qpos[:3].copy()
        root_height = root_pos[2]
        
        # Extract root velocities for velocity tracking rewards
        root_vel = self.data.qvel[:6].copy()  # 3 linear vel, 3 angular vel
        vx, vy, vz = root_vel[0], root_vel[1], root_vel[2]  # Linear velocities
        wx, wy, wz = root_vel[3], root_vel[4], root_vel[5]  # Angular velocities
        
        # Extract roll and pitch information for roll/pitch stabilization
        root_quat = self.data.qpos[3:7].copy()
        
        # Calculate roll and pitch from quaternion
        w, x, y, z = root_quat
        roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
        pitch = np.arcsin(2.0 * (w*y - z*x))
        
        # Get upright alignment from quaternion
        upright_alignment = 1.0 - 2.0 * (x*x + y*y)  # Simplified dot product calculation
        
        # Check for quaternion to matrix conversion
        if hasattr(mujoco, 'mju_quat2Mat'):
            rot_mat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rot_mat, root_quat)
            # Extract the z-axis (up vector) from rotation matrix
            up_vector = rot_mat[6:9]  # last column of rotation matrix
            # Compute dot product with world up [0,0,1]
            upright_alignment = up_vector[2]  # z component should be close to 1
            
        # Get position of the torso/body
        torso_id = -1
        for i in range(self.model.nbody):
            if hasattr(self.model, 'body_names'):
                # Check by name
                if "torso" in str(self.model.body_names[i]):
                    torso_id = i
                    break
            else:
                # Fall back to searching by position if body names aren't accessible
                body_pos = self.data.xpos[i]
                if 0.5 < body_pos[2] < 0.8:  # Height range for torso
                    torso_id = i
                    break
                
        if torso_id >= 0:
            # Get position of torso
            torso_pos = self.data.xpos[torso_id].copy()
        else:
            # Fallback to root position if torso not found
            torso_pos = root_pos
            
        # Check foot contacts using the optimized method
        foot_contact = self._check_foot_contacts()
        
        # Calculate rewards and penalties
        
        # 1. Linear Velocity Tracking Reward
        vxy_ref = np.array([self.vx_ref, self.vy_ref])
        vxy = np.array([vx, vy])
        lin_vel_tracking_reward = np.exp(-np.sum(np.square(vxy_ref - vxy)))
        lin_vel_tracking_reward = self.lin_vel_tracking_weight * lin_vel_tracking_reward
        
        # 2. Angular Velocity Tracking Reward
        ang_vel_tracking_reward = np.exp(-np.square(self.wz_ref - wz))
        ang_vel_tracking_reward = self.ang_vel_tracking_weight * ang_vel_tracking_reward
        
        # 3. Height Penalty
        height_penalty = -self.height_penalty_weight * np.square(root_height - self.z_ref)
        
        # 4. Pose Similarity Reward
        current_joint_pos = self.data.qpos[joint_start_idx:joint_start_idx+actual_joints_dim].copy()
        pose_similarity_penalty = -self.pose_similarity_weight * np.sum(np.square(current_joint_pos - self.default_joint_pos))
        
        # 5. Vertical Velocity Penalty 
        vertical_vel_penalty = -self.vertical_vel_penalty_weight * np.square(vz)
        
        # 6. Roll and Pitch Stabilization Penalty
        roll_pitch_penalty = -self.roll_pitch_penalty_weight * (np.square(roll) + np.square(pitch))
        orientation_penalty = -self.orientation_penalty_weight * (abs(roll) + abs(pitch))
        
        # 7. Velocity penalty (discourage excessive velocities)
        velocity_penalty = -self.velocity_penalty_weight * np.sum(np.square(self.data.qvel))
                    
        # 8. Contact reward (only if robot is upright)
        contact_reward = self.contact_reward_weight if foot_contact and upright_alignment > 0.8 else 0.0
            
        # 9. Torque penalty (based on action magnitude)
        torque_penalty = -self.torque_penalty_weight * np.sum(np.square(raw_action))
        
        # 10. Joint limit penalty
        joint_limit_penalty = -self.joint_limit_penalty_weight * joint_limit_violations
        
        # 11. Forward lean penalty (we want the robot to stay upright)
        lean_penalty = -self.forward_lean_penalty_weight * (1.0 - upright_alignment)**2
        
        # Compute overall reward based on all components
        # Base survival reward (for staying alive)
        reward = 0.3
        
        # Stability reward (using quadratic penalty for dropping below threshold)
        z0 = self.nominal_height
        height_diff = max(0, z0 - root_height)
        stability_reward = -self.stability_reward_weight * (height_diff ** 2)
        
        # Add bonus for staying upright and at a good height
        if root_height > self.nominal_height * 0.9 and upright_alignment > 0.9:
            stability_reward += self.stability_reward_weight * 0.5
        
        reward += stability_reward
        
        # Goal-reaching reward - only if robot is upright
        if upright_alignment > 0.8:
            goal_distance = np.linalg.norm(torso_pos[:2] - self.goal_pos[:2])
            
            # Goal distance component
            goal_reward = self.goal_reward_weight * (1.0 - min(goal_distance / 2.0, 1.0))
            
            # Distance progress component 
            prev_goal_distance = np.linalg.norm(prev_root_pos[:2] - self.goal_pos[:2])
            distance_progress = prev_goal_distance - goal_distance
            goal_progress_reward = self.goal_reward_weight * 5.0 * distance_progress
            
            # Extra bonus for being very close to goal
            if goal_distance < 0.1:
                goal_reward += self.goal_reward_weight * 2.0
                
            goal_reward = goal_reward + goal_progress_reward
        else:
            # If not upright, no goal reward
            goal_distance = float('inf')
            goal_reward = 0.0
        
        reward += goal_reward
        
        # Add all other reward components
        reward_components = [
            contact_reward,
            lean_penalty,
            velocity_penalty,
            torque_penalty,
            joint_limit_penalty,
            lin_vel_tracking_reward,
            ang_vel_tracking_reward,
            height_penalty,
            pose_similarity_penalty,
            action_rate_penalty,
            vertical_vel_penalty,
            roll_pitch_penalty,
            orientation_penalty
        ]
        
        reward += sum(reward_components)

        # Check for goal success (robot is close enough to goal AND upright)
        goal_success = goal_distance < self.goal_success_threshold and upright_alignment > 0.8
        
        # Check for termination (robot fell or reached goal)
        # Consider a fall when height is below threshold OR robot is severely tilted
        fall = root_height < self.fall_threshold or upright_alignment < 0.5
        terminated = fall or goal_success
        
        # Apply early termination penalty for falling
        if fall:
            reward -= self.early_termination_penalty
        
        # Apply goal success bonus
        if goal_success:
            reward += self.goal_reward_weight * 10.0  # Big bonus for reaching goal
        
        # Maximum episode length exceeded
        truncated = self.steps >= self.episode_length
        
        # Additional info with new reward components
        info = {
            'reward_stability': stability_reward,
            'reward_goal': goal_reward,
            'reward_contact': contact_reward,
            'penalty_torque': torque_penalty,
            'penalty_joint_limit': joint_limit_penalty,
            'penalty_velocity': velocity_penalty,
            'penalty_lean': lean_penalty,
            'reward_lin_vel_tracking': lin_vel_tracking_reward,
            'reward_ang_vel_tracking': ang_vel_tracking_reward,
            'penalty_height': height_penalty,
            'penalty_pose_similarity': pose_similarity_penalty,
            'penalty_action_rate': action_rate_penalty,
            'penalty_vertical_vel': vertical_vel_penalty,
            'penalty_roll_pitch': roll_pitch_penalty,
            'penalty_orientation': orientation_penalty,
            'root_height': root_height,
            'upright_alignment': upright_alignment,
            'torso_pos': torso_pos,
            'goal_distance': goal_distance,
            'goal_success': goal_success,
            'fall': fall,
            'foot_contact': foot_contact,
            'linear_velocity': np.array([vx, vy, vz]),
            'angular_velocity': np.array([wx, wy, wz]),
            'roll': roll,
            'pitch': pitch,
            'is_truncated': truncated,
            'is_terminated': terminated
        }
        
        # Render if in human mode
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            import glfw
            # Update scene and render
            self.renderer.update_scene(self.data)
            _ = self.renderer.render()
            # Present rendered image to window
            glfw.make_context_current(self.window)
            glfw.swap_buffers(self.window)
            # Poll events
            glfw.poll_events()
            # Query window size (required by some backends)
            glfw.get_window_size(self.window)
            # Check for window close
            if glfw.window_should_close(self.window):
                self.close()
                
        return None
        
    def close(self):
        """Clean up resources"""
        if self.viewer:
            import glfw
            glfw.terminate()
            self.viewer = None

class GymCompatibilityWrapper(gym.Wrapper):
    """Wrapper to handle API differences between the old Gym and new Gymnasium"""
    def __init__(self, env):
        super().__init__(env)
        # Add max episode steps for compatibility with PPO
        self._max_episode_steps = env.episode_length if hasattr(env, 'episode_length') else 1000
        
    def reset(self, **kwargs):
        """Handle both old and new gym APIs"""
        result = self.env.reset(**kwargs)
        # Gymnasium API returns (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # Old Gym API returns just obs
        return result, {}
    
    def step(self, action):
        """Handle both old and new gym APIs"""
        result = self.env.step(action)
        # Make sure we're returning (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            return result
        # Old Gym API returns (obs, reward, done, info)
        obs, reward, done, info = result
        return obs, reward, done, False, info
    
    def seed(self, seed=None):
        """Set random seed for environment"""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        # For Gymnasium compatibility
        logger.info(f"Using reset(seed={seed}) instead of seed() method")
        self.reset(seed=seed)
        return [seed]

def make_g1_env(**kwargs):
    """Create a custom configured G1 environment with appropriate parameters.
    
    This function creates a G1 environment with the updated parameters:
    - Goal distance: 3.0 meters
    - Strong upright orientation reward
    - Appropriate time limits for the 3m goal
    
    Args:
        render_mode (str): 'human' for rendering, None for headless
        seed (int): Random seed for reproducibility
        fixed_dims (bool): If True, use fixed dimensions to match expert buffer
        **kwargs: Additional keyword arguments for G1EnvConfig
    
    Returns:
        GymCompatibilityWrapper: A wrapped G1 environment
    """
    # Try to import and use the official G1 gym environment if available
    try:
        import gym
        import g1_gym
        logger.info("Using official G1 Gym environment")
        
        # Create the environment with default settings
        env = gym.make('G1Raw-v0', render_mode=kwargs.get('render_mode', None))
        
        # Set the goal distance to 3.0 meters
        if hasattr(env.unwrapped, 'goal_distance'):
            logger.info(f"Setting goal distance to 3.0 meters (previously {env.unwrapped.goal_distance})")
            env.unwrapped.goal_distance = 3.0
        
        # Increase upright orientation reward weight if available
        if hasattr(env.unwrapped, 'upright_weight'):
            prev_weight = env.unwrapped.upright_weight
            env.unwrapped.upright_weight = 0.5  # Stronger weight for upright orientation
            logger.info(f"Setting upright orientation weight to 0.5 (previously {prev_weight})")
        
        # Ensure max episode steps is set
        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = 1000  # Increased time limit for 3m navigation
            logger.info(f"Setting max episode steps to 1000")
        
        return env
        
    except ImportError:
        logger.info("G1 Gym environment not found, using direct G1Env implementation")
        
        # Create a config with default settings, overridden by any kwargs
        config = G1EnvConfig()
        
        # Override config with any provided kwargs that match config attributes
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Always set goal position consistently
        config.goal_pos = np.array([0.0, -1.0, 0.7])  # 3.0m to the side (Y-axis)
        logger.info(f"Setting goal distance to 3.0 meters (Y-axis)")
        
        # Use direct G1Env implementation
        env = G1Env(render_mode=kwargs.get('render_mode', None), config=config)
        
        # Wrap for compatibility if needed
        wrapped_env = GymCompatibilityWrapper(env)
        
        return wrapped_env

# For testing the environment directly
if __name__ == "__main__":
    # Set up console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # Create environment with specified parameters
    env = make_g1_env(
        render_mode=None,
        stability_reward_weight=4.0,
        forward_lean_penalty_weight=2.0
    )
    
    # Reset with a specific seed
    seed = 42
    obs, info = env.reset(seed=seed)
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Action space: {env.action_space}")
    
    total_reward = 0
    episode_length = 0
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        
        if (i + 1) % 20 == 0:
            logger.info(f"Step {i+1}: Reward={reward:.3f}, Total={total_reward:.3f}")
        
        if terminated or truncated:
            logger.info(f"Episode ended after {episode_length} steps with total reward {total_reward:.3f}")
            logger.info(f"Reason: {'Success' if info.get('goal_success', False) else 'Fall' if info.get('fall', False) else 'Timeout'}")
            break
            
    env.close()
    logger.info("Environment test completed.") 