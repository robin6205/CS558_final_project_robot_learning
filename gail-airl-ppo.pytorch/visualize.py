import os
import argparse
import torch
import numpy as np
import time

from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.buffer import SerializedBuffer

def run(args):
    # Create environment with human rendering mode
    env = make_env(args.env_id, render_mode="human")
    
    # Create and load the algorithm
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=torch.device("cuda" if args.cuda else "cpu")
    )
    
    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length
    )
    
    # Load weights
    algo.actor.load_state_dict(torch.load(args.model_path))
    print(f"Model loaded from {args.model_path}")
    
    # Run a few episodes for visualization
    for episode in range(args.num_episodes):
        state = env.reset(seed=args.seed + episode)
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            action = algo.exploit(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Add a small delay to observe the movement better
            time.sleep(0.01)
            
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {episode_steps}")
    
    env.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='InvertedPendulum-v2')
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--rollout_length', type=int, default=2000)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_episodes', type=int, default=5)
    args = p.parse_args()
    run(args) 