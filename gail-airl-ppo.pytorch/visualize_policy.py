import os
import time
import argparse
import torch
import numpy as np
import mujoco
import sys

# Check for mujoco_viewer
try:
    import mujoco_viewer
except ImportError:
    print("Error: mujoco_viewer not found.")
    print("Please install with: pip install mujoco-python-viewer")
    sys.exit(1)

# Import policy model - update to use the history-enabled policy
from StateIndependentGaussianPolicy import StateIndependentGaussianPolicy

# Get the absolute path of the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(args):
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    print(f"Loading policy from {args.model_path}")
    
    # Resolve model path
    model_path = args.model_path
    if not model_path.endswith(".pth"):
        model_path = os.path.join(model_path, "actor.pth")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Load MuJoCo model directly (like in the working script)
    xml_filepath = os.path.join(BASE_DIR, "data", "g1_robot", "g1_23dof_simplified.xml")
    
    try:
        # Load the MuJoCo model and data
        print(f"Loading MuJoCo model from {xml_filepath}")
        model = mujoco.MjModel.from_xml_path(xml_filepath)
        data = mujoco.MjData(model)
        print(f"Successfully loaded model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
        
        # Adjust friction parameters to prevent slipping
        for contact_id in range(model.nconmax):
            # Set sliding friction (higher = more friction)
            model.pair_friction[contact_id, 0] = 2.0  # Increase from 1.0 to 2.0
            # Set torsional friction
            model.pair_friction[contact_id, 1] = 0.2  # Increase from 0.1 to 0.2
            # Set rolling friction
            model.pair_friction[contact_id, 2] = 0.2  # Increase from 0.1 to 0.2
        print("Friction parameters adjusted")
        
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        sys.exit(1)
    
    # Get state and action dimensions from the model
    state_dim = 46  # 23 joint angles + 23 velocities
    action_dim = model.nu  # Number of actuators
    
    # Initialize policy network with history support
    policy = StateIndependentGaussianPolicy(
        state_shape=(state_dim,),
        action_shape=(action_dim,),
        hidden_units=(64, 64),
        hidden_activation=torch.nn.Tanh(),
        history_length=args.history_length
    ).to(device)
    
    # Load policy weights
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    print(f"Policy loaded successfully with history_length={args.history_length}")
    
    # For debugging - print a small part of model weights
    for name, param in policy.named_parameters():
        print(f"Parameter {name} shape: {param.shape}")
        if param.numel() > 5:
            print(f"First 5 values: {param.data.flatten()[:5]}")
        break
    
    # Set initial state 
    initial_qpos = np.array([
        0, 0, 0.79,  # Base position (x, y, z)
        1, 0, 0, 0,  # Base orientation (w, x, y, z)
        # Legs (L: P, R, Y, Knee, AP, AR | R: P, R, Y, Knee, AP, AR)
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        # Waist (Y)
        0,
        # Arms (L: SP, SR, SY, Elbow, WR | R: SP, SR, SY, Elbow, WR)
        0.2, 0.2, 0, 1.28, 0,
        0.2, -0.2, 0, 1.28, 0
    ])
    
    # Setup MuJoCo viewer
    print("Setting up MuJoCo viewer...")
    try:
        viewer = mujoco_viewer.MujocoViewer(model, data, title="G1 Robot Policy Visualization")
    except Exception as e:
        print(f"Error setting up viewer: {e}")
        sys.exit(1)
    
    # Run episodes
    for ep in range(1, args.num_episodes + 1):
        print(f"\nStarting episode {ep}")
        
        # Reset simulation state
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)
        
        # Reset policy's state history
        if hasattr(policy, 'state_history'):
            policy.state_history = torch.zeros_like(policy.state_history)
        
        # Set goal position further away from the initial position
        initial_pos = data.qpos[:3].copy()
        goal_y = initial_pos[1] + 0.4  # 0.4m ahead (changed from 0.5)
        goal_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        if goal_site_id >= 0:
            data.site_pos[goal_site_id] = np.array([0.0, goal_y, 0.1])  # Update goal position
        
        print(f"Goal set at y={goal_y:.2f} (initial y={initial_pos[1]:.2f})")
        
        total_reward = 0.0
        step = 0
        done = False
        goal_reached = False
        goal_success_threshold = 0.27  # Changed from 0.3 to 0.27 to match our env
        
        while not done and step < args.max_steps and viewer.is_alive:
            try:
                # Get the observation (23 joint angles + 23 velocities)
                joint_angles = data.qpos[-23:].copy()  # Last 23 entries are the joint angles
                joint_vels = data.qvel[-23:].copy()    # Last 23 entries are the joint velocities
                
                # Combine into observation
                obs = np.concatenate([joint_angles, joint_vels])
                
                # Get roll and pitch for display
                quat = data.qpos[3:7].copy()  # Quaternion orientation
                w, x, y, z = quat
                roll = np.arctan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y))
                pitch = np.arcsin(2.0 * (w*y - z*x))
                
                # Get velocities for tracking display
                vx = data.qvel[0]
                vy = data.qvel[1]
                vz = data.qvel[2]
                wz = data.qvel[5]  # Yaw velocity
                
                # Use policy to get action
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    # Sample from the policy
                    action_tensor, _ = policy.sample(obs_tensor)
                action = action_tensor.cpu().numpy().squeeze()
                
                # Apply action to joint actuators
                data.ctrl[:] = action
                
                # Save position for reward calculation
                prev_root_pos = data.qpos[:3].copy()
                
                # Step simulation (run for a few steps to stabilize)
                for _ in range(5):
                    mujoco.mj_step(model, data)
                
                # Calculate reward
                new_root_pos = data.qpos[:3].copy()
                root_height = new_root_pos[2]
                
                # Get torso position (more reliable than root for goal distance)
                torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                if torso_id >= 0:
                    # Get position of torso
                    torso_pos = data.xpos[torso_id].copy()
                else:
                    # Fallback to root position if torso not found
                    torso_pos = new_root_pos
                
                # Goal distance
                prev_goal_dist = np.linalg.norm(prev_root_pos[:2] - np.array([0.0, goal_y]))
                goal_distance = np.linalg.norm(torso_pos[:2] - np.array([0.0, goal_y]))
                
                # Calculate tracking rewards
                vx_ref = 0.5  # Desired forward velocity (matching training)
                vy_ref = 0.0
                wz_ref = 0.0
                lin_vel_tracking = np.exp(-np.sum(np.square(np.array([vx_ref, vy_ref]) - np.array([vx, vy]))))
                ang_vel_tracking = np.exp(-np.square(wz_ref - wz))
                
                # Goal progress reward
                goal_progress = prev_goal_dist - goal_distance  # Positive when getting closer
                goal_reward_weight = 10.0  # Strong incentive to reach goal
                goal_reward = goal_reward_weight * goal_progress
                
                # Stability reward
                stability_height_threshold = 0.5
                stability_reward_weight = 0.8
                if root_height > stability_height_threshold:
                    stability_reward = stability_reward_weight
                else:
                    stability_reward = root_height / stability_height_threshold * stability_reward_weight
                
                # Base survival reward
                base_reward = 0.3
                
                # Roll/pitch stability reward (higher is better)
                roll_pitch_reward = 1.0 - (roll**2 + pitch**2) * 0.5  # Normalized to [0,1]
                roll_pitch_reward = max(0, roll_pitch_reward)  # Ensure non-negative
                
                # Combine rewards
                reward = base_reward + stability_reward + goal_reward
                
                # Check for goal success
                goal_success = goal_distance < goal_success_threshold
                if goal_success and not goal_reached:
                    goal_reached = True
                    print(f"Goal reached at step {step}!")
                    reward += 20.0  # Bonus for reaching goal
                
                total_reward += reward
                step += 1
                
                # Check termination
                fall_threshold = 0.4
                fall = bool(root_height < fall_threshold)
                done = fall or goal_reached
                
                # Render scene
                viewer.render()
                
                # Print step info (every 10 steps)
                if step % 10 == 0 or done:
                    status = "GOAL!" if goal_reached else ("FALL!" if fall else "")
                    print(f"Step {step}: Reward={reward:.2f}, Height={root_height:.2f}, Goal dist={goal_distance:.2f} {status}")
                    print(f"  Roll/Pitch: ({roll:.2f}, {pitch:.2f}), Vel: ({vx:.2f}, {vy:.2f}, {vz:.2f})")
                    print(f"  Tracking: lin_vel={lin_vel_tracking:.2f}, ang_vel={ang_vel_tracking:.2f}")
                
                # Control playback speed
                time.sleep(1.0 / args.fps)
                
            except Exception as e:
                print(f"Error during step {step}: {e}")
                break
        
        episode_result = "SUCCESS" if goal_reached else ("FALL" if fall else "TIMEOUT")
        print(f"Episode {ep} complete: {episode_result} - Reward = {total_reward:.3f}, Steps = {step}")
        
        # Small pause between episodes
        time.sleep(1.0)
    
    # Clean up
    if viewer.is_alive:
        viewer.close()
    
    print("Visualization complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a trained GAIL policy on the G1 robot")
    parser.add_argument('--model_path', type=str, required=True, help='Path to actor.pth or model directory')
    parser.add_argument('--num_episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=30, help='Rendering frame rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--history_length', type=int, default=1, help='History length to match trained policy')
    args = parser.parse_args()
    
    try:
        run(args)
    except Exception as e:
        import traceback
        print(f"Error in visualization: {e}")
        traceback.print_exc() 