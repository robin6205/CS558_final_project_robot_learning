import pandas as pd
import torch
import numpy as np
import os
import argparse
from g1_env import make_g1_env
from scipy.signal import savgol_filter

def make_buffer(csv_path, output_path, time_col='Timestamp', exclude_cols=None, smooth_window=9, smooth_order=3):
    """
    Reads humanoid motion data from a CSV file, processes it into state-action pairs,
    and saves it as a PyTorch buffer compatible with the imitation learning setup.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path where the output .pth buffer file will be saved.
        time_col (str): Name of the timestamp column in the CSV (to exclude from state data)
        exclude_cols (list): Additional column names to exclude from the state
        smooth_window (int): Window size for Savitzky-Golay smoothing filter (must be odd)
        smooth_order (int): Polynomial order for smoothing filter
    """
    # Define joint angle scaling factors to reduce extreme movements
    joint_angle_scale = {
        # --- Hip ---
        'LeftHip_pitch': 0.2,
        'RightHip_pitch': 0.2,

        'LeftHip_roll': 0.2,
        'RightHip_roll': 0.2,

        'LeftHip_yaw': 0.5,
        'RightHip_yaw': 0.5,

        # --- Knee ---
        'LeftKnee_flexion': 0.2,
        'RightKnee_flexion': 0.2,

        # --- Ankle ---
        'LeftAnkle_pitch': 0.2,
        'RightAnkle_pitch': 0.2,

        'LeftAnkle_roll': 0.2,
        'RightAnkle_roll': 0.2,

        # --- Shoulder ---
        'LeftShoulder_pitch': 0.3,
        'RightShoulder_pitch': 0.3,

        'LeftShoulder_roll': 0.3,
        'RightShoulder_roll': 0.3,

        'LeftShoulder_yaw': 0.5,
        'RightShoulder_yaw': 0.5,

        # --- Elbow ---
        'LeftElbow_flexion': 0.3,
        'RightElbow_flexion': 0.3,

        # --- Wrist ---
        'LeftWrist_pronation': 0.0,  # Minimal wrist movement
        'RightWrist_pronation': 0.0,

        # --- Waist ---
        'Waist_yaw': 0.3
    }

    print(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Get all columns except timestamp and any other specified excluded columns
    exclude_cols = exclude_cols or []
    if time_col:
        exclude_cols.append(time_col)
    
    # All joint angle columns
    joint_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not joint_cols:
        print(f"Error: No joint angle columns found after excluding {exclude_cols}")
        return
    
    print(f"Found {len(joint_cols)} joint angle columns: {joint_cols}")

    # Extract joint angles (qpos) and timestamps
    qpos_raw = df[joint_cols].values  # Shape T x num_joints
    try:
        timestamps = df[time_col].values  # Shape T
    except KeyError:
        print(f"Error: Timestamp column '{time_col}' not found in CSV.")
        return
        
    # Create a copy for the scaled and smoothed version
    qpos = qpos_raw.copy()
    
    # Apply scaling to each joint angle column
    for idx, col in enumerate(joint_cols):
        if col not in joint_angle_scale:
            print(f"Warning: No scaling factor defined for joint column '{col}'. Using default scale = 0.5.")
            scale = 0.5  # Conservative default scale to reduce movement
        else:
            scale = joint_angle_scale[col]

        # Scale the joint angle values
        qpos[:, idx] *= scale
        print(f"Applied scale {scale} to joint column: {col}")
    
    # Apply smoothing to reduce jitter in the joint angles
    try:
        # Ensure smooth_window is odd for Savitzky-Golay filter
        if smooth_window % 2 == 0:
            smooth_window += 1
            
        if qpos.shape[0] > smooth_window:  # Only smooth if we have enough data points
            print(f"Applying Savitzky-Golay smoothing with window={smooth_window}, order={smooth_order}")
            for j in range(qpos.shape[1]):
                qpos[:, j] = savgol_filter(qpos[:, j], smooth_window, smooth_order)
        else:
            print(f"Not enough data points for smoothing with window size {smooth_window}")
    except Exception as e:
        print(f"Error during smoothing: {e}. Continuing without smoothing.")

    num_joints = qpos.shape[1]
    print(f"Number of joints detected: {num_joints}")

    if qpos.shape[0] < 2:
        print("Error: Need at least two timesteps in the CSV to compute actions and velocities.")
        return

    # Calculate dt and qvel (joint velocities)
    dt_values = np.diff(timestamps)
    if np.any(dt_values <= 0):
        invalid_dts = dt_values[dt_values <= 0]
        print(f"Warning: Found non-positive time differences: {invalid_dts}. Check timestamp column '{time_col}'.")
        # Option: Replace invalid dt with average? Or raise error? For now, use average of positives.
        positive_dts = dt_values[dt_values > 0]
        if len(positive_dts) == 0:
             print("Error: All time differences are non-positive. Cannot calculate velocities. Using default dt=1/30.")
             dt = 1.0 / 30.0 # Default if no valid dt found
        else:
             dt = np.mean(positive_dts)
             print(f"Using average positive dt: {dt:.4f}")
    else:
        dt = np.mean(dt_values)
        print(f"Average timestep dt: {dt:.4f}")

    if dt <= 1e-6: # Check if dt is too small or zero
         print(f"Warning: Calculated dt is very small or zero ({dt}). Using default dt=1/30.")
         dt = 1.0 / 30.0

    # Calculate joint changes and velocities
    dq = np.diff(qpos, axis=0)  # Change in joint angles. Shape (T-1) x num_joints
    
    # Scale down large movements to avoid extreme actions
    max_change_threshold = 0.1  # Maximum allowed change in radians per step
    large_changes = np.abs(dq) > max_change_threshold
    if np.any(large_changes):
        print(f"Warning: Found {np.sum(large_changes)} large joint changes. Scaling them down.")
        # Scale down large changes while preserving direction
        scale_factors = np.ones_like(dq)
        scale_factors[large_changes] = max_change_threshold / np.abs(dq[large_changes])
        dq = dq * scale_factors
            
    # Calculate velocities based on the changes
    qvel = dq / dt  # Joint velocities. Shape (T-1) x num_joints
    
    # Estimate velocity for the last frame T-1 by repeating the last calculated velocity
    # This makes qvel align with qpos timesteps: qvel[t] is velocity *at* timestep t
    qvel = np.vstack([qvel, qvel[-1:]])  # Shape T x num_joints

    # State s_t = [qpos_t, qvel_t] corresponds to action a_t, leading to s_{t+1}
    # states_t contains states from t=0 to T-2
    states_t = np.hstack([qpos[:-1], qvel[:-1]])  # Shape (T-1) x (2 * num_joints)
    # next_states_t contains states from t=1 to T-1
    next_states_t = np.hstack([qpos[1:], qvel[1:]])  # Shape (T-1) x (2 * num_joints)

    # Action a_t = delta qpos = qpos[t+1] - qpos[t] (unnormalized)
    actions_raw = dq  # Shape (T-1) x num_joints

    # Normalize actions using environment bounds
    actions_norm = actions_raw  # Initialize in case normalization fails
    print("Creating dummy G1 environment to get action scaling...")
    try:
        env = make_g1_env()
        # Check if action space is Box
        if not hasattr(env.action_space, 'high'):
             print("Warning: Environment action space does not have 'high' attribute. Cannot determine scale. Skipping action normalization.")
        else:
            scale = env.action_space.high  # Assuming this is numpy array, shape (num_joints,)
            if not isinstance(scale, np.ndarray):
                scale = np.array(scale)

            if scale.shape != (num_joints,):
                 print(f"Warning: Action scale shape {scale.shape} doesn't match action dimension {num_joints}. Skipping normalization.")
            else:
                # Avoid division by zero or very small numbers if bounds are zero/tiny
                scale[np.abs(scale) < 1e-6] = 1.0
                actions_norm = actions_raw / scale
                print(f"Applied action normalization using scale (env.action_space.high).")
                print(f"Action scale range: [{np.min(scale)}, {np.max(scale)}]")
                print(f"Normalized action range: [{np.min(actions_norm)}, {np.max(actions_norm)}]")

        # Clean up the environment if possible (some envs might have a close method)
        if hasattr(env, 'close'):
            env.close()
    except ImportError:
        print("Error: Could not import make_g1_env. Make sure g1_env.py is accessible.")
        print("Skipping action normalization.")
    except Exception as e:
        print(f"Error getting action scale from environment: {e}")
        print("Skipping action normalization.")

    # Final clipping of actions to ensure they're within the normalized range (-1, 1)
    actions_norm = np.clip(actions_norm, -0.9, 0.9)  # Leave small margin for safety
    
    num_transitions = states_t.shape[0]
    print(f"Generated {num_transitions} transitions.")
    print(f"State dimension: {states_t.shape[1]}")
    print(f"Action dimension: {actions_norm.shape[1]}")

    # Create dummy rewards and dones
    rewards_t = np.zeros((num_transitions, 1), dtype=np.float32)
    # Assume it's one long episode, only the last transition leads to a 'done' state
    dones_t = np.zeros((num_transitions, 1), dtype=np.bool_)
    dones_t[-1] = True
    
    # Convert to PyTorch tensors
    states_tensor = torch.tensor(states_t, dtype=torch.float32)
    actions_tensor = torch.tensor(actions_norm, dtype=torch.float32)
    rewards_tensor = torch.tensor(rewards_t, dtype=torch.float32)
    dones_tensor = torch.tensor(dones_t, dtype=torch.bool)
    next_states_tensor = torch.tensor(next_states_t, dtype=torch.float32)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    # Save the buffer
    buffer_data = {
        'state': states_tensor,
        'action': actions_tensor,
        'reward': rewards_tensor,
        'done': dones_tensor,
        'next_state': next_states_tensor
    }
    
    try:
        torch.save(buffer_data, output_path)
        print(f"Expert buffer saved successfully to {output_path}")
        print("Buffer content shapes:")
        print(f"  States:      {states_tensor.shape}")
        print(f"  Actions:     {actions_tensor.shape}")
        print(f"  Rewards:     {rewards_tensor.shape}")
        print(f"  Dones:       {dones_tensor.shape}")
        print(f"  Next States: {next_states_tensor.shape}")
    
    except Exception as e:
        print(f"Error saving buffer to {output_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate expert buffer from CSV motion data.")
    parser.add_argument('--csv', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--out', type=str, default='buffers/side_step_expert.pth', help='Path to save the output buffer file.')
    parser.add_argument('--time_col', type=str, default='Timestamp', help='Name of timestamp column to exclude')
    parser.add_argument('--exclude', nargs='+', default=[], help='Additional columns to exclude')
    parser.add_argument('--smooth_window', type=int, default=9, help='Window size for smoothing (must be odd number)')
    parser.add_argument('--smooth_order', type=int, default=3, help='Polynomial order for smoothing')
    args = parser.parse_args()

    make_buffer(args.csv, args.out, args.time_col, args.exclude, args.smooth_window, args.smooth_order)

    # Verification step
    print("\nTo verify, run:")
    print(f"python -c \"import torch; expert = torch.load('{args.out}'); print(expert['state'].shape, expert['action'].shape); print('Action range:', expert['action'].min().item(), expert['action'].max().item())\"") 