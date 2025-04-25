import argparse
import time
import torch
import mujoco_viewer
from g1_env import G1Env
from gail_airl_ppo.network.policy import StateIndependentPolicy


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained GAIL policy on the G1 robot")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved actor.pth')
    parser.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--fps', type=float, default=30.0, help='Render frame rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed offset')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Create environment without built-in rendering
    env = G1Env(render_mode=None)

    # Load policy
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    policy = StateIndependentPolicy(state_shape=obs_shape, action_shape=act_shape).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()

    # Setup MuJoCo viewer
    viewer = mujoco_viewer.MujocoViewer(env.model, env.data)

    # Run episodes
    for ep in range(1, args.num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        print(f"Episode {ep} start")
        while not done and step < args.max_steps and viewer.is_alive:
            # Sample action
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_tensor).cpu().numpy()[0]

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1

            # Render simulation
            viewer.render()
            time.sleep(1.0 / args.fps)

        print(f"Episode {ep} complete: Reward = {total_reward:.3f}, Steps = {step}")

    # Cleanup
    if viewer.is_alive:
        viewer.close()
    env.close()


if __name__ == '__main__':
    main() 