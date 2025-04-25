import gym
import logging

# Set gym logger level
logging.getLogger('gym').setLevel(logging.ERROR)


def make_env(env_id, render_mode=None):
    # For newer versions of environments with new versions
    if env_id == "InvertedPendulum-v2":
        env_id = "InvertedPendulum-v4" 
    elif env_id == "Hopper-v3":
        env_id = "Hopper-v4"
    # Allow our custom G1-v0 environment
    elif env_id == "G1-v0":
        # Make sure g1_env is imported to register the environment
        import sys
        import os
        # Add project root to path if needed
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import g1_env
        # Use our compatibility wrapper
        return NormalizedEnv(g1_env.make_g1_env())
    
    return NormalizedEnv(gym.make(env_id))


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        # Extract max_episode_steps if available
        try:
            self._max_episode_steps = env.spec.max_episode_steps
        except AttributeError:
            # Try underlying env if it's a wrapper
            spec = getattr(getattr(env, 'env', None), 'spec', None)
            self._max_episode_steps = spec.max_episode_steps if spec is not None else None

        self.scale = env.action_space.high
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale

    def step(self, action):
        # Handle both old and new gym API formats
        step_result = self.env.step(action * self.scale)
        
        # New Gym API returns (obs, reward, terminated, truncated, info)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            return obs, reward, terminated, truncated, info
        # Old Gym API returns (obs, reward, done, info)
        else:
            obs, reward, done, info = step_result
            return obs, reward, done, info
        
    def reset(self):
        # Handle both old and new gym API formats
        reset_result = self.env.reset()
        
        # New API returns (obs, info)
        if isinstance(reset_result, tuple) and len(reset_result) >= 1:
            return reset_result
        # Old API returns just obs
        else:
            return reset_result
