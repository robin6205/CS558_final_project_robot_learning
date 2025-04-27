import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# Get base directory for consistent file references
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class G1Env(gym.Env):
    """
    Custom Gym environment for the Unitree G1 robot
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        # Env parameters
        self.frame_skip = 5
        # Set goal position for side-step to the right
        self.goal_pos = np.array([1.5, 0.0, 0.0])  # 1.5m to the right (X-axis)
        self.goal_success_threshold = 0.25  # 25cm threshold for success
        self.goal_reward_weight = 5.0  # Weight for goal-directed reward
        
        # Updated stability parameters
        self.stability_reward_weight = 2.0  # Increased weight for stability
        self.nominal_height = 0.55  # Expected standing height
        self.fall_threshold = 0.35  # Robot is considered fallen below this height
        self.early_termination_penalty = 10.0  # Severe penalty for falling
        
        # New reward components
        self.contact_reward_weight = 1.0  # Weight for foot contact
        self.torque_penalty_weight = 0.001  # Weight for torque penalty
        self.joint_limit_penalty_weight = 0.05  # Weight for joint limits penalty
        self.forward_lean_penalty_weight = 2.0  # Penalty for leaning forward/backward
        self.velocity_penalty_weight = 0.01  # Penalty for excessive velocities
        
        # Additional new reward components
        self.lin_vel_tracking_weight = 1.0  # Weight for linear velocity tracking
        self.ang_vel_tracking_weight = 1.0  # Weight for angular velocity tracking
        self.height_penalty_weight = 1.5  # Weight for height penalty
        self.pose_similarity_weight = 0.5  # Weight for pose similarity
        self.action_rate_penalty_weight = 0.01  # Weight for action rate penalty
        self.vertical_vel_penalty_weight = 1.0  # Weight for vertical velocity penalty
        self.roll_pitch_penalty_weight = 2.0  # Weight for roll/pitch stabilization
        
        # Default commanded velocities and height (can be updated)
        self.vx_ref = 0.0  # Default target x velocity
        self.vy_ref = 0.0  # Default target y velocity
        self.wz_ref = 0.0  # Default target yaw velocity
        self.z_ref = self.nominal_height  # Default target height
        
        # Store previous actions for action rate penalty
        self.prev_action = None
        
        # Default joint positions (standing pose)
        self.default_joint_pos = np.zeros(23)  # Will be set in reset
        
        # Environment transition parameters
        self.action_scale = 0.02  # Reduced action scale for better stability
        self.episode_length = 500  # Shorter episodes for faster learning
        
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
        
        print(f"Loaded model with qpos dimension: {self.qpos_dim}, qvel dimension: {self.qvel_dim}")
        
        # Define action space (joint angle changes) - reduced scale for stability
        n_dof = 23
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_dof,), dtype=np.float32
        )
        
        # Define observation space based on our trimmed observation (23 joint angles + 23 velocities)
        # This matches the expert data dimensions
        obs_dim = 46  # 23 joint angles + 23 velocities
        print(f"Using observation dimension: {obs_dim} (trimmed to match expert data)")
        
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
        self.foot_geom_ids = []
        self.foot_site_ids = []
        
        # Find foot contact geoms
        for i in range(self.model.ngeom):
            # Try to find foot geoms by name if available
            if hasattr(self.model, 'geom_names') and i < len(self.model.geom_names):
                geom_name = str(self.model.geom_names[i])
                if 'foot' in geom_name or 'ankle' in geom_name:
                    self.foot_geom_ids.append(i)
            # Otherwise use heuristics (spheres with small size near the ground)
            elif self.model.geom_type[i] == 2:  # Type 2 is sphere in MuJoCo
                if self.model.geom_size[i, 0] < 0.05:  # Small spheres
                    self.foot_geom_ids.append(i)
        
        # Use foot site IDs if available
        for i in range(self.model.nsite):
            # Try to find by name
            if hasattr(self.model, 'site_names') and i < len(self.model.site_names):
                site_name = str(self.model.site_names[i])
                if 'foot' in site_name or 'ankle' in site_name:
                    self.foot_site_ids.append(i)
            else:
                # Add all sites as potential contact points
                self.foot_site_ids.append(i)
            
        print(f"Found {len(self.foot_geom_ids)} foot contact geoms and {len(self.foot_site_ids)} foot sites")
    
    def seed(self, seed=None):
        self.np_random.seed(seed)
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
        
        # In MJCF, qpos includes:
        # - 3 values for the root position (x, y, z)
        # - 4 values for the root orientation (quaternion)
        # - The rest are joint angles
        # We only want the joint angles (last 23 elements) to match expert data
        if len(qpos) > 23:
            qpos_joints = qpos[-23:]  # Take only the last 23 joint values
        else:
            qpos_joints = qpos  # Keep all if already right size
            
        # Similarly for qvel
        if len(qvel) > 23:
            qvel_joints = qvel[-23:]  # Take only the last 23 velocity values
        else:
            qvel_joints = qvel  # Keep all if already right size
        
        # Concatenate to get the full state and convert to float32
        obs = np.concatenate([qpos_joints, qvel_joints]).astype(np.float32)
        
        # Replace any NaN or Inf values with zeros
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs
        
    def reset(self, seed=None, options=None):
        """Reset the environment to a random initial state"""
        if seed is not None:
            self.seed(seed)
        
        # Reset the simulation and preserve root position & orientation from XML
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
        joint_start_idx = self.qpos_dim - 23
        joint_start_idx = max(7, joint_start_idx)
        self.default_joint_pos = self.data.qpos[joint_start_idx:joint_start_idx+23].copy()
        
        # Reset previous action
        self.prev_action = np.zeros(23)
        
        # Forward dynamics to get the simulation into a valid state
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial height for comparison
        self.initial_height = self.data.qpos[2]
        
        # Get observation
        obs = self._get_obs()
        
        # Return observation and empty info dict
        return obs, {}
    
    def step(self, action):
        """Step the simulation forward based on the action"""
        self.steps += 1
        
        # Scale and clip action for stability
        action = np.clip(action, -1.0, 1.0) * self.action_scale
        
        # Record the raw action for torque penalty calculation
        raw_action = action.copy()
        
        # Calculate action rate penalty if we have a previous action
        action_rate_penalty = 0.0
        if self.prev_action is not None:
            action_diff = raw_action - self.prev_action
            action_rate_penalty = -self.action_rate_penalty_weight * np.sum(np.square(action_diff))
        
        # Update previous action for next step
        self.prev_action = raw_action.copy()
        
        # We can only control the actual joint DOFs, not the root position/orientation
        # Skip the first 7 qpos values (3 for position, 4 for quaternion orientation)
        # and apply action to the last 23 joint values (if model has more)
        joint_start_idx = self.qpos_dim - 23  # Calculate where the last 23 joints start
        joint_start_idx = max(7, joint_start_idx)  # Ensure we're at least past the root
        
        # Apply actions to the joints
        n_action = min(len(action), 23)  # We expect 23 actions
        
        # Record root position before simulation for reward calculation
        prev_root_pos = self.data.qpos[:3].copy()
        prev_joint_pos = self.data.qpos[joint_start_idx:joint_start_idx+n_action].copy()
        
        # Apply actions with more careful stepping
        for i in range(n_action):
            self.data.qpos[joint_start_idx + i] += action[i]
        
        # Add joint limit constraint to prevent extreme values
        joint_limits = 3.14  # approx pi, reasonable joint limit in radians
        joint_pos = self.data.qpos[joint_start_idx:joint_start_idx+n_action]
        
        # Store values for joint limit penalty calculation
        upper_limits = np.ones_like(joint_pos) * joint_limits
        lower_limits = -upper_limits
        
        # Calculate joint limit violations for penalty
        joint_limit_violations = np.sum(
            np.maximum(0, joint_pos - upper_limits)**2 + 
            np.maximum(0, lower_limits - joint_pos)**2
        )
        
        # Apply clipping to enforce joint limits
        self.data.qpos[joint_start_idx:joint_start_idx+n_action] = np.clip(
            self.data.qpos[joint_start_idx:joint_start_idx+n_action],
            -joint_limits, joint_limits
        )

        # Run the simulation with smaller steps for better stability
        try:
            for _ in range(self.frame_skip):  # Take more smaller steps 
                # Check if state is valid before stepping
                if np.any(np.isnan(self.data.qpos)) or np.any(np.isinf(self.data.qpos)):
                    print("Warning: Invalid state detected (NaN/Inf in qpos)")
                    # Reset to previous valid state
                    self.data.qpos[:] = np.nan_to_num(self.data.qpos, nan=0.0, posinf=0.0, neginf=0.0)
                    
                # Step with smaller timestep for stability
                mujoco.mj_step2(self.model, self.data)  # Just run kinematics
                
                # Add damping to velocities for stability
                self.data.qvel[:] *= 0.98  # More damping (was 0.99)
                
                # Run dynamics
                mujoco.mj_step1(self.model, self.data)
        except Exception as e:
            print(f"Simulation error: {e}")
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
        
        # Check orientation (we want the robot to remain upright)
        # Extract quaternion of the root (indices 3,4,5,6)
        # Convert quaternion to rotation matrix
        # Simplified method for checking upright orientation:
        # The 'up' vector in world coordinates is [0,0,1]
        # When transformed by the rotation, we want it to remain mostly vertical
        
        # Calculate roll and pitch from quaternion
        # This is a simplified approach to extract approximate roll and pitch
        w, x, y, z = root_quat
        roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
        pitch = np.arcsin(2.0 * (w*y - z*x))
        
        # Check if we have the quaternion to matrix conversion function
        if hasattr(mujoco, 'mju_quat2Mat'):
            rot_mat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rot_mat, root_quat)
            
            # Extract the z-axis (up vector) from rotation matrix
            up_vector = rot_mat[6:9]  # last column of rotation matrix
            
            # Compute dot product with world up [0,0,1]
            # (should be close to 1 if robot is upright)
            upright_alignment = up_vector[2]  # z component should be close to 1
        else:
            # Simplified check using just the quaternion
            # This assumes the identity orientation has the z-axis aligned with world up
            w, x, y, z = root_quat
            upright_alignment = 1.0 - 2.0 * (x*x + y*y)  # simplified dot product calculation
            
        # Get position of the torso/body (more reliable than root for goal distance)
        torso_id = -1
        for i in range(self.model.nbody):
            if hasattr(self.model, 'body_names'):
                # Check if body_names attribute exists (newer MuJoCo versions)
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
            
        # Check foot contacts - at least one foot should be on the ground
        foot_contact = False
        
        # Check for contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in self.foot_geom_ids or contact.geom2 in self.foot_geom_ids:
                foot_contact = True
                break
        
        # Alternative check using foot sites - check if they're close to the ground
        if not foot_contact and len(self.foot_site_ids) > 0:
            for site_id in self.foot_site_ids:
                site_pos = self.data.site_xpos[site_id]
                if site_pos[2] < 0.1:  # Site is close to ground
                    foot_contact = True
                    break
                    
        # NEW REWARD COMPONENTS:
        
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
        current_joint_pos = self.data.qpos[joint_start_idx:joint_start_idx+n_action].copy()
        pose_similarity_penalty = -self.pose_similarity_weight * np.sum(np.square(current_joint_pos - self.default_joint_pos))
        
        # 5. Action Rate Penalty was calculated earlier
        
        # 6. Vertical Velocity Penalty 
        vertical_vel_penalty = -self.vertical_vel_penalty_weight * np.square(vz)
        
        # 7. Roll and Pitch Stabilization Penalty
        roll_pitch_penalty = -self.roll_pitch_penalty_weight * (np.square(roll) + np.square(pitch))
        
        # Calculate velocity penalty (discourage excessive velocities)
        velocity_penalty = -self.velocity_penalty_weight * np.sum(np.square(self.data.qvel))
                    
        # Calculate contact reward (only if robot is upright)
        contact_reward = self.contact_reward_weight if foot_contact and upright_alignment > 0.8 else 0.0
            
        # Calculate torque penalty (based on action magnitude)
        torque_penalty = -self.torque_penalty_weight * np.sum(np.square(raw_action))
        
        # Calculate joint limit penalty
        joint_limit_penalty = -self.joint_limit_penalty_weight * joint_limit_violations
        
        # Forward lean penalty (we want the robot to stay upright)
        lean_penalty = -self.forward_lean_penalty_weight * (1.0 - upright_alignment)**2
        
        # Compute reward based on multiple components
        # 1. Base survival reward (for staying alive)
        reward = 0.3
        
        # 2. Stability reward (using quadratic penalty for dropping below threshold)
        z0 = self.nominal_height
        height_diff = max(0, z0 - root_height)
        stability_reward = -self.stability_reward_weight * (height_diff ** 2)
        
        # Add bonus for staying upright and at a good height
        if root_height > self.nominal_height * 0.9 and upright_alignment > 0.9:
            stability_reward += self.stability_reward_weight * 0.5
        
        reward += stability_reward
        
        # 3. Goal-reaching reward - only if robot is upright
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
        
        # 4. Add contact, lean, velocity, torque and joint limit components
        reward += contact_reward
        reward += lean_penalty
        reward += velocity_penalty  
        reward += torque_penalty
        reward += joint_limit_penalty
        
        # 5. Add new reward components
        reward += lin_vel_tracking_reward
        reward += ang_vel_tracking_reward
        reward += height_penalty
        reward += pose_similarity_penalty
        reward += action_rate_penalty
        reward += vertical_vel_penalty
        reward += roll_pitch_penalty

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
            'pitch': pitch
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

class GymCompatibilityWrapper:
    """Wrapper to handle API differences between the old Gym and new Gymnasium"""
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # Add max episode steps for compatibility with PPO
        self._max_episode_steps = env.episode_length if hasattr(env, 'episode_length') else 1000
        # Expose the unwrapped environment
        self.unwrapped = env
        
    def reset(self, **kwargs):
        """Handle both old and new gym APIs"""
        result = self.env.reset(**kwargs)
        # New Gymnasium API returns (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            return result  # Just return the (obs, info) tuple
        # Old Gym API returns just obs
        return result, {}  # Return (obs, {}) for compatibility
        
    def step(self, action):
        """Handle both old and new gym APIs"""
        result = self.env.step(action)
        # New Gymnasium API returns (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            return result
        # Old Gym API returns (obs, reward, done, info)
        obs, reward, done, info = result
        return obs, reward, done, False, info  # Return (obs, reward, terminated, truncated, info)
        
    def render(self, mode="human"):
        """Handle both old and new gym render APIs"""
        try:
            return self.env.render()
        except TypeError:
            return self.env.render(mode=mode)
            
    def close(self):
        return self.env.close()

def make_g1_env(**kwargs):
    """Create a custom configured G1 environment with appropriate parameters.
    
    This function creates a G1 environment with the updated parameters:
    - Goal distance: 3.0 meters
    - Strong upright orientation reward
    - Appropriate time limits for the 3m goal
    """
    try:
        import gym
        import g1_gym
    except ImportError:
        print("G1 Gym environment not found, using direct G1Env implementation")
        # Use direct G1Env implementation
        env = GymCompatibilityWrapper(G1Env(render_mode=kwargs.get('render_mode', None)))
        
        # Set the goal distance to 3.0 meters
        if hasattr(env.unwrapped, 'goal_pos'):
            print(f"Setting goal distance to 3.0 meters")
            env.unwrapped.goal_pos = np.array([3.0, 0.0, 0.0])  # 3.0m to the right (X-axis)
        
        # Increase upright orientation reward weight if available
        if hasattr(env.unwrapped, 'forward_lean_penalty_weight'):
            env.unwrapped.forward_lean_penalty_weight = 2.0
            print(f"Setting upright penalty weight to 2.0")
        
        # Increase time limit for the 3-meter goal
        if hasattr(env, '_max_episode_steps'):
            env._max_episode_steps = 1000  # Increased time limit for 3m navigation
            print(f"Setting max episode steps to 1000")
        
        return env
    
    # Create the environment with default settings (original code for when g1_gym is available)
    env = gym.make('G1Raw-v0', **kwargs)
    
    # Set the goal distance to 3.0 meters
    if hasattr(env.unwrapped, 'goal_distance'):
        print(f"Setting goal distance to 3.0 meters (previously {env.unwrapped.goal_distance})")
        env.unwrapped.goal_distance = 3.0
    
    # Increase upright orientation reward weight if available
    if hasattr(env.unwrapped, 'upright_weight'):
        prev_weight = env.unwrapped.upright_weight
        env.unwrapped.upright_weight = 0.5  # Stronger weight for upright orientation
        print(f"Setting upright orientation weight to 0.5 (previously {prev_weight})")
    
    # Increase time limit for the 3-meter goal if needed
    if hasattr(env, 'max_episode_steps'):
        env._max_episode_steps = 1000  # Increased time limit for 3m navigation
        print(f"Setting max episode steps to 1000")
    
    print(f"G1 environment created with 3.0m goal distance and enhanced upright orientation reward")
    return env

# For testing the environment directly
if __name__ == "__main__":
    env = make_g1_env()
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Action shape: {action.shape}")
        print(f"Observation shape: {obs.shape}")
        print(f"Reward: {reward}")
        
        if done:
            break
            
    env.close()
    print("Environment test completed.") 