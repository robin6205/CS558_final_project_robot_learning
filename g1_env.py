import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import pickle

# Register environment with gymnasium if available
try:
    from gymnasium.envs.registration import register
    try:
        register(
            id='G1Raw-v0',
            entry_point='g1_env:G1Env',
            max_episode_steps=1000,
        )
    except Exception:
        # Environment might already be registered
        pass
except ImportError:
    # Try older gym registration
    try:
        from gym.envs.registration import register
        try:
            register(
                id='G1Raw-v0',
                entry_point='g1_env:G1Env',
                max_episode_steps=1000,
            )
        except Exception:
            # Environment might already be registered
            pass
    except ImportError:
        pass

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
        self.goal_pos = np.array([0.0, 0.25, 0.0])  # 25cm away in the y-direction
        self.goal_success_threshold = 0.25  # 25cm threshold for success
        self.goal_reward_weight = 10.0
        self.stability_reward_weight = 3.0  # Increased stability reward weight
        self.stability_height_threshold = 0.5
        self.fall_threshold = 0.38  # Slightly lower fall threshold to give more time
        self.action_scale = 0.03
        self.episode_length = 1000
        
        # Tracking variables
        self.steps = 0
        self.render_mode = render_mode
        
        # Load the model
        model_path = os.path.join(BASE_DIR, "data/g1_robot/g1_23dof_simplified.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
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
        
        # Neutralize only joint DOFs (preserve first 7 qpos for root)
        n_root = 7  # 3 for root pos, 4 for root quaternion
        self.data.qpos[n_root:] = 0.0
        self.data.qvel[:] = 0.0  # Reset all velocities to zero
        
        # Add a small amount of noise to joint positions for exploration
        joint_noise = self.np_random.uniform(-0.01, 0.01, size=self.qpos_dim-n_root)
        self.data.qpos[n_root:] += joint_noise
        
        # Forward dynamics to get the simulation into a valid state
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial height for comparison
        self.initial_height = self.data.qpos[2]
        
        # Set goal position at 25cm away in the y-direction
        initial_pos = self.data.qpos[:3].copy()
        self.goal_pos = np.array([initial_pos[0], initial_pos[1] + 0.25, initial_pos[2]])
        self.goal_y_pos = self.goal_pos[1]
        print(f"Setting goal at y={self.goal_pos[1]:.2f} (initial y={initial_pos[1]:.2f})")
        
        # Reset tracking variables
        self.steps = 0
        
        # Get the observation
        obs = self._get_obs()
        
        # Extra info
        info = {
            'goal_pos': self.goal_pos.copy(),
            'initial_height': self.initial_height
        }
        
        return obs, info
    
    def step(self, action):
        """Step the simulation forward based on the action"""
        self.steps += 1
        
        # Scale and clip action for stability
        action = np.clip(action, -1.0, 1.0) * self.action_scale
        
        # We can only control the actual joint DOFs, not the root position/orientation
        # Skip the first 7 qpos values (3 for position, 4 for quaternion orientation)
        # and apply action to the last 23 joint values (if model has more)
        joint_start_idx = self.qpos_dim - 23  # Calculate where the last 23 joints start
        joint_start_idx = max(7, joint_start_idx)  # Ensure we're at least past the root
        
        # Apply actions to the joints
        n_action = min(len(action), 23)  # We expect 23 actions
        
        # Record root position before simulation for reward calculation
        prev_root_pos = self.data.qpos[:3].copy()
        
        # Apply actions with more careful stepping
        for i in range(n_action):
            self.data.qpos[joint_start_idx + i] += action[i]
        
        # Add joint limit constraint to prevent extreme values
        joint_limits = 3.14  # approx pi, reasonable joint limit in radians
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
                self.data.qvel[:] *= 0.99  # Slight damping
                
                # Run dynamics
                mujoco.mj_step1(self.model, self.data)
        except Exception as e:
            print(f"Simulation error: {e}")
            # Return early with terminated=True if simulation fails
            return self._get_obs(), 0.0, True, False, {"error": str(e)}

        # Get the new observation
        obs = self._get_obs()

        # Get state information
        pos = self.data.qpos[:3].copy()
        root_height = pos[2]

        # Compute reward based on multiple components
        # 1. Base survival reward
        reward = 0.5
        
        # 2. Stability reward (positive for staying upright)
        if root_height > self.stability_height_threshold:
            stability_reward = self.stability_reward_weight
        else:
            stability_reward = (root_height / self.stability_height_threshold) * self.stability_reward_weight
        reward += stability_reward
        
        # 3. Goal-reaching reward - more progressive reward structure
        goal_distance = np.linalg.norm(pos[:2] - self.goal_pos[:2])
        
        # Progressive goal reward:
        # - High reward when very close (within 5cm)
        # - Moderate reward when making progress (5-15cm)
        # - Smaller reward when further away
        if goal_distance < 0.05:
            goal_reward = self.goal_reward_weight * 3.0
        elif goal_distance < 0.15:
            goal_reward = self.goal_reward_weight * 1.5
        else:
            # Smooth decay based on distance
            goal_reward = self.goal_reward_weight * (1.0 - min(goal_distance / 0.3, 1.0))
        
        # Add reward for y-direction progress (towards goal)
        y_progress = pos[1] - prev_root_pos[1]
        if y_progress > 0 and pos[1] <= self.goal_pos[1]:  # Only reward progress towards goal, not past it
            progress_reward = y_progress * 15.0  # Increased reward for forward movement
            reward += progress_reward
        
        reward += goal_reward

        # Check for goal success (robot is close enough to goal)
        goal_success = goal_distance < self.goal_success_threshold

        # Check for termination (robot fell or reached goal)
        fall = root_height < self.fall_threshold
        terminated = fall or goal_success
        
        # Maximum episode length exceeded
        truncated = self.steps >= self.episode_length
        
        # Additional info
        info = {
            'reward_stability': stability_reward,
            'reward_goal': goal_reward,
            'root_height': root_height,
            'goal_distance': goal_distance,
            'goal_success': goal_success,
            'fall': fall,
            'goal_pos': self.goal_pos.copy()
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

# Register the environment
from gymnasium.envs.registration import register

# Register the raw environment
register(
    id='G1Raw-v0',
    entry_point='g1_env:G1Env',
    max_episode_steps=1000,
)

# Compatibility wrapper to handle API differences
class GymCompatibilityWrapper:
    """Wrapper to ensure compatibility between gym and gymnasium APIs"""
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self):
        try:
            # Gymnasium API (returns tuple of obs and info)
            result = self.env.reset()
            if isinstance(result, tuple) and len(result) == 2:
                return result[0]  # Return just the observation for older gym API
            return result
        except TypeError:
            # Fall back to older gym API
            return self.env.reset()
    
    def step(self, action):
        result = self.env.step(action)
        # Handle both 4-tuple (obs, reward, done, info) and 5-tuple (obs, reward, term, trunc, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        return result
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def close(self):
        return self.env.close()

# Create a custom make function for compatibility
def make_g1_env(render_mode=None):
    """Create a G1 environment wrapped for gym compatibility"""
    env = G1Env(render_mode=render_mode)
    return GymCompatibilityWrapper(env)

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