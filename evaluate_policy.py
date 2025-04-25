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

def evaluate_policy(env_id, model_path, render=True, episodes=5, seed=0):
    # Create environment
    print(f"Creating G1 environment...")
    env = make_g1_env()
    
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
    
    # Evaluation metrics
    total_reward = 0
    success_count = 0
    fall_count = 0
    total_steps = 0
    avg_distance = 0
    avg_final_height = 0
    
    # Evaluate for some episodes
    for ep in range(episodes):
        print(f"\nEpisode {ep+1}/{episodes}")
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        episode_success = False
        episode_fall = False
        
        while not done and step < 1000:  # Add step limit to prevent infinite loops
            # Use the trained policy to choose an action
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                action = actor(state_tensor).detach().cpu().numpy()[0]
            
            # Take the action in the environment
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Render the environment if requested
            if render:
                try:
                    env.render()
                except Exception as e:
                    print(f"Error rendering: {e}")
                    render = False  # Disable rendering if it fails
            
            # Debug info at every step
            if isinstance(info, dict):
                height = info.get('root_height', 0)
                goal_distance = info.get('goal_distance', float('inf'))
                goal_success = info.get('goal_success', False)
                fall = info.get('fall', False)
                
                # Track success and fall
                if goal_success:
                    episode_success = True
                if fall:
                    episode_fall = True
                
                # Print detailed info every step, regardless of count
                print(f"Step {step}: Reward: {reward:.2f}, Height: {height:.2f}, Goal dist: {goal_distance:.2f}")
                print(f"  Success: {goal_success}, Fall: {fall}, Done: {done}")
                
                # If specific reward components are available, show them
                if 'reward_lateral' in info:
                    print(f"  Reward breakdown - Lateral: {info['reward_lateral']:.2f}, " +
                          f"Goal: {info.get('reward_goal', 0):.2f}, " +
                          f"Stability: {info.get('reward_stability', 0):.2f}")
        
        # Update metrics
        total_reward += episode_reward
        total_steps += step
        if episode_success:
            success_count += 1
        if episode_fall:
            fall_count += 1
        
        # Get final metrics
        if isinstance(info, dict):
            avg_distance += info.get('goal_distance', 0)
            avg_final_height += info.get('root_height', 0)
        
        # Report episode results
        success_text = "SUCCESS!" if episode_success else ("FALL" if episode_fall else "TIMEOUT")
        print(f"Episode {ep+1} completed with reward: {episode_reward:.2f} in {step} steps - {success_text}")
    
    # Calculate averages
    average_reward = total_reward / episodes
    success_rate = success_count / episodes * 100
    fall_rate = fall_count / episodes * 100
    avg_steps = total_steps / episodes
    avg_distance /= episodes
    avg_final_height /= episodes
    
    # Print overall evaluation results
    print(f"\n===== EVALUATION RESULTS =====")
    print(f"Average reward over {episodes} episodes: {average_reward:.2f}")
    print(f"Success rate: {success_count}/{episodes} episodes ({success_rate:.1f}%)")
    print(f"Fall rate: {fall_count}/{episodes} episodes ({fall_rate:.1f}%)")
    print(f"Average episode length: {avg_steps:.1f} steps")
    print(f"Average final distance to goal: {avg_distance:.3f}")
    print(f"Average final robot height: {avg_final_height:.3f}")
    print(f"============================")
    
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