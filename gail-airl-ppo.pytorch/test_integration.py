import os
import sys

# Add the project path so we can import gail_airl_ppo
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

from gail_airl_ppo.env import make_env

print("Testing G1 environment integration...")

# Create the G1 environment using the make_env function
env = make_env('G1-v0')
obs = env.reset()

print(f"Environment created successfully!")
print(f"Observation shape: {obs.shape}")

# Test a few steps
for i in range(3):
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    print(f"Step {i+1}: Action shape: {action.shape}, Observation shape: {next_obs.shape}")

env.close()
print("Integration test successful!") 