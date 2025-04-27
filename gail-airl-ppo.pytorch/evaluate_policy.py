import os
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from collections import defaultdict

from g1_env import make_g1_env
from StateIndependentGaussianPolicy import StateIndependentGaussianPolicy


def evaluate_policy(env, policy, num_episodes, render=False, deterministic=False, action_repeat=1, delay=0.0, history_length=1):
    """
    Evaluate a policy for a specified number of episodes.
    
    Args:
        env: Environment
        policy: Policy to evaluate
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        action_repeat: Number of times to repeat each action
        delay: Delay between frames when rendering (in seconds)
        history_length: Number of previous states to consider
    
    Returns:
        Dictionary with evaluation statistics
    """
    device = next(policy.parameters()).device
    
    rewards = []
    episode_lengths = []
    distances = []
    uprights = []
    success_steps = []
    successes = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        success = False
        
        # Reset policy history at the beginning of each episode
        if hasattr(policy, 'state_history'):
            policy.state_history = torch.zeros_like(policy.state_history)
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                if deterministic:
                    if hasattr(policy, 'forward'):
                        action = policy.forward(state_tensor)
                    elif hasattr(policy, 'mode'):
                        action = policy.mode(state_tensor)
                    else:
                        action = policy(state_tensor)
                else:
                    if hasattr(policy, 'sample'):
                        action, _ = policy.sample(state_tensor)
                    else:
                        action = policy(state_tensor)
                action = action.cpu().numpy().flatten()
            
            # Repeat action for specified number of times
            for _ in range(action_repeat):
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                
                # Render if required
                if render:
                    env.render()
                    if delay > 0:
                        time.sleep(delay)
                
                # Check if goal reached - success if distance < 0.5
                if 'goal_distance' in info and info['goal_distance'] < 0.5:
                    success = True
                    if not done:  # If success but episode not terminated yet
                        done = True
                
                state = next_state
                
                if done:
                    break
        
        # Track metrics
        rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        # Extract final state information
        final_distance = info.get('goal_distance', np.nan)
        final_upright = info.get('upright_alignment', np.nan)
        distances.append(final_distance)
        uprights.append(final_upright)
        
        if success:
            successes += 1
            success_steps.append(steps)
        
        print(f"Episode {ep+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}, "
              f"Distance = {final_distance:.2f}m, Upright = {final_upright:.2f}, Success = {success}")
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_steps = np.mean(episode_lengths)
    std_steps = np.std(episode_lengths)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    mean_upright = np.mean(uprights)
    std_upright = np.std(uprights)
    success_rate = (successes / num_episodes) * 100
    
    # Calculate success statistics if there were any successes
    if successes > 0:
        mean_success_steps = np.mean(success_steps)
        std_success_steps = np.std(success_steps)
    else:
        mean_success_steps = np.nan
        std_success_steps = np.nan
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_steps': mean_steps,
        'std_steps': std_steps,
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'mean_upright': mean_upright,
        'std_upright': std_upright,
        'success_rate': success_rate,
        'mean_success_steps': mean_success_steps,
        'std_success_steps': std_success_steps,
        'rewards': rewards,
        'episode_lengths': episode_lengths,
        'distances': distances,
        'uprights': uprights,
        'success_steps': success_steps
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="G1Raw-v0", help="environment name")
    parser.add_argument("--model-path", type=str, required=True, help="path to model")
    parser.add_argument("--episodes", type=int, default=10, help="number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument("--action-repeat", type=int, default=1, help="number of times to repeat the same action")
    parser.add_argument("--history-length", type=int, default=1, help="number of previous states to consider")
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = make_g1_env(render_mode="human" if args.render else None)
    print(f"Created environment: {args.env}")
    
    # Load model
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    # Create policy with Gaussian distribution
    model = StateIndependentGaussianPolicy(
        state_shape=state_shape, 
        action_shape=action_shape,
        hidden_units=(64, 64),
        hidden_activation=torch.nn.Tanh(),
        history_length=args.history_length
    )
    
    # Load the model weights
    model_weights = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_weights)
    model.eval()
    
    print(f"Loaded model from: {args.model_path}")
    print(f"Using state history length: {args.history_length}")
    
    # Evaluate policy
    results = evaluate_policy(
        env=env,
        policy=model,
        num_episodes=args.episodes,
        render=args.render,
        deterministic=False,  # Use sampling for stochastic policy
        action_repeat=args.action_repeat,
        history_length=args.history_length
    )
    
    # Print summary of results
    print("\n===== Evaluation Results =====")
    print(f"Episodes: {args.episodes}")
    print(f"Average reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Average episode length: {results['mean_steps']:.2f} ± {results['std_steps']:.2f}")
    print(f"Success rate: {results['success_rate']:.1f}%")
    print(f"Average final distance: {results['mean_distance']:.2f}m ± {results['std_distance']:.2f}m")
    print(f"Average final upright: {results['mean_upright']:.2f} ± {results['std_upright']:.2f}")
    
    if results['success_rate'] > 0:
        print(f"Average steps to success: {results['mean_success_steps']:.2f} ± {results['std_success_steps']:.2f}") 