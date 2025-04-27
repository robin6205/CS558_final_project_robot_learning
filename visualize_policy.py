import argparse
import time
import torch
import mujoco_viewer
import sys
sys.path.append("gail-airl-ppo.pytorch")
from g1_env import G1Env
from StateIndependentGaussianPolicy import StateIndependentGaussianPolicy


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained GAIL policy on the G1 robot")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved actor.pth')
    parser.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--fps', type=float, default=30.0, help='Render frame rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed offset')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--history_length', type=int, default=1, help='Number of previous states to consider')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Create environment without built-in rendering
    env = G1Env(render_mode=None)

    # Load policy
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    
    # Create the Gaussian policy model with history support
    policy = StateIndependentGaussianPolicy(
        state_shape=obs_shape, 
        action_shape=act_shape,
        hidden_units=(64, 64),
        hidden_activation=torch.nn.Tanh(),
        history_length=args.history_length
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    
    print(f"Loaded policy from {args.model_path}")
    print(f"Using state history length: {args.history_length}")

    # Setup MuJoCo viewer
    viewer = mujoco_viewer.MujocoViewer(env.model, env.data)

    # Run episodes
    for ep in range(1, args.num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        
        # Reset policy history at the start of each episode
        if hasattr(policy, 'state_history'):
            policy.state_history = torch.zeros_like(policy.state_history)
        
        print(f"Episode {ep} start")
        
        # Track metrics for this episode
        min_goal_distance = float('inf')
        final_distance = None
        final_upright = None
        success = False
        
        while not done and step < args.max_steps and viewer.is_alive:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Sample action
            with torch.no_grad():
                if hasattr(policy, 'sample'):
                    action, _ = policy.sample(obs_tensor)
                    action = action.cpu().numpy().flatten()
                else:
                    action = policy(obs_tensor).cpu().numpy().flatten()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # Track metrics
            if 'goal_distance' in info:
                distance = info['goal_distance']
                min_goal_distance = min(min_goal_distance, distance)
                final_distance = distance
                # Check for success - goal reached if distance < 0.5
                if distance < 0.5:
                    success = True
                    print(f"Goal reached! Distance: {distance:.2f}")
                    
            if 'upright_alignment' in info:
                final_upright = info['upright_alignment']
            
            # Display real-time stats
            if step % 10 == 0:
                distance_str = f", Distance: {final_distance:.2f}m" if final_distance is not None else ""
                upright_str = f", Upright: {final_upright:.2f}" if final_upright is not None else ""
                print(f"Step {step}: Reward: {reward:.2f}{distance_str}{upright_str}")

            # Render simulation
            viewer.render()
            time.sleep(1.0 / args.fps)

        # Episode summary
        success_str = "SUCCESS!" if success else "Failed"
        print(f"Episode {ep} complete: {success_str}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps: {step}")
        print(f"Min Goal Distance: {min_goal_distance:.2f}m")
        print(f"Final Distance: {final_distance:.2f}m")
        print(f"Final Upright: {final_upright:.2f}")
        print("-" * 40)

    # Cleanup
    if viewer.is_alive:
        viewer.close()
    env.close()


if __name__ == '__main__':
    main() 