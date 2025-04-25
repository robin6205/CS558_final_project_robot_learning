import os
import argparse
import torch
import time
import numpy as np

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import ALGOS

def run(args):
    # Find the latest model if not specified
    if args.model_path is None:
        # Look in the logs directory for the most recent model
        logs_dir = os.path.join('logs', args.env_id, args.algo)
        if not os.path.exists(logs_dir):
            print(f"Error: Logs directory {logs_dir} does not exist.")
            return
            
        # Find the latest run directory
        runs = sorted(os.listdir(logs_dir))
        if not runs:
            print(f"Error: No runs found in {logs_dir}.")
            return
            
        latest_run = os.path.join(logs_dir, runs[-1])
        model_dir = os.path.join(latest_run, 'model')
        
        # Find the latest step directory
        steps = sorted([d for d in os.listdir(model_dir) if d.startswith('step')])
        if not steps:
            print(f"Error: No model steps found in {model_dir}.")
            return
            
        latest_step = os.path.join(model_dir, steps[-1])
        args.model_path = os.path.join(latest_step, 'actor.pth')
        print(f"Using latest model: {args.model_path}")

    # Create environment with rendering
    env = make_env(args.env_id, render_mode="human")
    
    # Load the algorithm (only need the actor for evaluation)
    # Initialize in a way that's compatible with the GAIL algorithm structure
    algo = ALGOS[args.algo](
        buffer_exp=None,  # Not needed for evaluation
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=100  # Doesn't matter for evaluation
    )
    
    # Load just the actor (policy) network weights if the file exists
    if os.path.exists(args.model_path):
        print(f"Loading actor model from {args.model_path}")
        try:
            algo.actor.load_state_dict(torch.load(args.model_path))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will use randomly initialized policy instead")
    else:
        print(f"Warning: Model file {args.model_path} not found. Using random policy.")
    
    # Run episodes for visualization
    total_rewards = []
    
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        print(f"\nEpisode {episode+1}:")
        
        while not done:
            # Use the trained policy (or random if model failed to load)
            action = algo.exploit(state)
            
            # Take a step in the environment
            state, reward, done, info = env.step(action)
            
            # Update stats
            episode_reward += reward
            episode_steps += 1
            
            # Add a small delay to better visualize the movement
            time.sleep(0.01)
            
            # Print periodic updates
            if episode_steps % 10 == 0:
                print(f"Step {episode_steps}: Reward = {reward:.3f}, Episode Reward = {episode_reward:.3f}")
        
        print(f"Episode {episode+1} complete: Total Reward = {episode_reward:.3f}, Steps = {episode_steps}")
        total_rewards.append(episode_reward)
    
    # Print summary statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nSummary: Avg Reward = {avg_reward:.3f}, Total Episodes = {args.num_episodes}")
    
    env.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='G1-v0', help='Environment ID')
    p.add_argument('--algo', type=str, default='gail', help='Algorithm (gail or airl)')
    p.add_argument('--model_path', type=str, help='Path to the actor model file (if None, will use latest)')
    p.add_argument('--cuda', action='store_true', help='Use GPU')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    p.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to evaluate')
    args = p.parse_args()
    
    run(args) 