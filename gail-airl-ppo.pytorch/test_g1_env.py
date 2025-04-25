import gymnasium as gym
import g1_env  # Import to register the environment

# Test the environment
env = gym.make('G1-v0')
obs, _ = env.reset()
for _ in range(5):
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
print("OK", obs.shape)
env.close() 