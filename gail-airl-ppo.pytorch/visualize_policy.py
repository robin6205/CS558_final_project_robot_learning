#!/usr/bin/env python3
import argparse
import time

import torch
import numpy as np
from g1_env import make_g1_env

# import your exact ActorCritic from the training script
from gail_ppo_train import ActorCritic  

def visualize(model_path: str, episodes: int, seed: int):
    # load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"]
    policy_state = ckpt["policy_state"]

    # create wrapped environment (with padding, truncation, correct dims)
    env = make_g1_env(render_mode="human")
    
    # seed everything
    torch.manual_seed(seed)
    np.random.seed(seed)

    # build policy network exactly as during training
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ac = ActorCritic(
        obs_dim, act_dim,
        hidden_sizes=args.hidden_sizes,
        learn_std=bool(getattr(args, "learn_std", False)),
        log_std_min=getattr(args, "log_std_min", -20.0),
        log_std_max=getattr(args, "log_std_max", 2.0),
    )
    ac.load_state_dict(policy_state)
    ac.eval()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        while not done:
            # forward pass
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                mu, std, _ = ac(obs_t)

            # deterministic action from mean, clipped and scaled
            a = mu.squeeze(0).cpu().numpy()
            a = np.clip(a, -1.0, 1.0) * env.unwrapped.action_scale

            # step and render
            obs, r, terminated, truncated, info = env.step(a)
            env.render()            # will only do something if render_mode="human"
            done = terminated or truncated
            total_r += r

            # slow down to real-time (~60 Hz)
            time.sleep(1 / env.metadata["render_fps"])

        print(f"Episode {ep+1:2d} return: {total_r:.2f}")

    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True, help="Path to best_model.pt or final_model.pt")
    p.add_argument("--episodes", type=int, default=5, help="How many episodes to run")
    p.add_argument("--seed",     type=int, default=0, help="Random seed")
    args = p.parse_args()
    visualize(args.model, args.episodes, args.seed)
