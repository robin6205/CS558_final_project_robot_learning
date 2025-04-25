import os
import argparse
import torch
import torch.nn as nn
from g1_env import make_g1_env

def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

class StateIndependentPolicy(nn.Module):
    """Implements the same actor architecture as in the original code"""
    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))
        
    def get_action(self, states):
        return self.forward(states).cpu().numpy()

def evaluate_policy(env_id, model_path, render=True, episodes=5, seed=0):
    # Create environment
    print(f"Creating G1 environment...")
    render_mode = "human" if render else None
    env = make_g1_env(render_mode=render_mode)
    
    # Set seed
    torch.manual_seed(seed)
    
    # Load the saved model
    print(f"Loading model from {model_path}...")
    
    # Get state dimension and action dimension from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create actor model with the same architecture as the original
    actor = StateIndependentPolicy(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        hidden_units=(64, 64),  # Match the original model's architecture
        hidden_activation=nn.Tanh()
    )
    
    # Look for the model files directly
    if os.path.exists(f"{model_path}/actor.pth"):
        actor_state_dict = torch.load(f"{model_path}/actor.pth")
        actor.load_state_dict(actor_state_dict)
        print("Actor model found and loaded")
    else:
        raise FileNotFoundError(f"Could not find actor model at {model_path}/actor.pth")
    
    # Set to evaluation mode
    actor.eval()
    device = torch.device("cpu")
    print("Model loaded successfully!")
    
    # Evaluate for some episodes
    total_reward = 0
    success_count = 0
    for ep in range(episodes):
        print(f"\nEpisode {ep+1}/{episodes}")
        # Run episode
        reset_result = env.reset()
        # Handle both old and new gym API
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
            
        episode_reward = 0
        episode_steps = 0
        episode_success = False
        
        while True:
            # Select action
            with torch.no_grad():
                action = actor.get_action(torch.FloatTensor(obs).to(device))
            
            # Execute action
            step_result = env.step(action)
            
            # Handle both old and new gym API
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
                terminated, truncated = done, False
            
            # Print debug info at each step
            print(f"Step {episode_steps+1}: Reward: {reward:.2f}, Height: {info.get('root_height', 0):.2f}, Goal dist: {info.get('goal_distance', 0):.2f}")
            print(f"  Robot pos: x={info.get('torso_pos', [0, 0, 0])[0]:.2f}, y={info.get('torso_pos', [0, 0, 0])[1]:.2f}, z={info.get('torso_pos', [0, 0, 0])[2]:.2f}")
            print(f"  Success: {info.get('goal_success', False)}, Fall: {info.get('fall', False)}, Done: {done}")
            
            episode_reward += reward
            episode_steps += 1
            
            # Check if episode is done
            if done:
                if info.get('goal_success', False):
                    episode_success = True
                break
            
            # Update observation
            obs = next_obs
        
        total_reward += episode_reward
        success_text = "SUCCESS!" if episode_success else "FAILED"
        print(f"Episode {ep+1} completed with reward: {episode_reward:.2f} in {episode_steps} steps - {success_text}")
        if episode_success:
            success_count += 1
    
    average_reward = total_reward / episodes
    success_rate = success_count / episodes * 100
    print(f"Average reward over {episodes} episodes: {average_reward:.2f}")
    print(f"Success rate: {success_count}/{episodes} episodes ({success_rate:.1f}%)")
    
    env.close()
    print("Evaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="G1-v0", help="Environment ID")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model files")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    args = parser.parse_args()
    
    evaluate_policy(
        env_id=args.env_id,
        model_path=args.model_dir,
        render=not args.no_render,
        episodes=args.episodes,
        seed=args.seed
    ) 