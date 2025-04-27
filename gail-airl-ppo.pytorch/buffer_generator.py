import os
import argparse
import pandas as pd
import numpy as np
import torch
from scipy.signal import butter, filtfilt, savgol_filter
from mujoco_py import load_model_from_path, MjSim
from g1_env import make_g1_env

# 1. Define mapping from mocap CSV columns to robot joint names
JOINT_MAPPING = {
    'left_hip_flexion': 'left_hip_pitch_joint',
    'left_hip_adduction': 'left_hip_roll_joint',
    'left_hip_rotation': 'left_hip_yaw_joint',
    'left_knee_flexion': 'left_knee_joint',
    'left_ankle_flexion': 'left_ankle_pitch_joint',
    'left_ankle_inversion': 'left_ankle_roll_joint',
    'right_hip_flexion': 'right_hip_pitch_joint',
    'right_hip_adduction': 'right_hip_roll_joint',
    'right_hip_rotation': 'right_hip_yaw_joint',
    'right_knee_flexion': 'right_knee_joint',
    'right_ankle_flexion': 'right_ankle_pitch_joint',
    'right_ankle_inversion': 'right_ankle_roll_joint',
}

# 2. Filtering utilities

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# 3. Load and preprocess mocap data with Butterworth + Savitzky-Golay smoothing
def load_and_preprocess(csv_path, joint_columns, lowpass_cutoff=6.0, fs=200, savgol_window=9, savgol_order=3):
    df = pd.read_csv(csv_path)
    for col in joint_columns:
        # Butterworth low-pass
        df[col] = lowpass_filter(df[col].values, lowpass_cutoff, fs)
        # Savitzky-Golay for jitter removal (window must be odd)
        if len(df[col]) > savgol_window:
            if savgol_window % 2 == 0:
                savgol_window += 1
            df[col] = savgol_filter(df[col].values, savgol_window, savgol_order)
    return df

# 4. Robust dt calculation from timestamps
def compute_dt(timestamps, default_dt=1/200.):
    dt_vals = np.diff(timestamps)
    pos = dt_vals[dt_vals > 0]
    if len(pos) > 0:
        dt = np.mean(pos)
    else:
        dt = default_dt
    return max(dt, default_dt)

# 5. Clip to MuJoCo joint limits
def enforce_limits(angle, limits):
    lo, hi = limits
    return np.clip(angle, lo, hi)

# 6. Main buffer creation
def create_hybrid_buffer(args):
    # Preprocess mocap
    joint_cols = list(JOINT_MAPPING.keys())
    df = load_and_preprocess(
        args.csv, joint_cols,
        lowpass_cutoff=args.lowpass,
        fs=args.fs,
        savgol_window=args.savgol_window,
        savgol_order=args.savgol_order
    )

    # Load MJ model and sim
    model = load_model_from_path(args.model_xml)
    sim = MjSim(model)
    # Joint limits
    joint_limits = {jn: tuple(model.jnt_range[model.joint_name2id(jn)])
                    for jn in JOINT_MAPPING.values()}

    # Retarget to qpos
    qpos_list = []
    timestamps = args.time_col and df[args.time_col].values if args.time_col in df else np.arange(len(df)) / args.fs

    for idx, row in df.iterrows():
        for mocap_col, joint_name in JOINT_MAPPING.items():
            angle = row[mocap_col]
            angle = enforce_limits(angle, joint_limits[joint_name])
            qpos_addr = model.get_joint_qpos_addr(joint_name)
            sim.data.qpos[qpos_addr] = angle
        # Step sim to settle
        sim.step()
        qpos_list.append(sim.data.qpos.copy())

    qpos_arr = np.stack(qpos_list)
    # Compute dt
    dt = compute_dt(df[args.time_col].values) if args.time_col in df else 1/args.fs

    # Compute dq, qvel
    dq = np.diff(qpos_arr, axis=0)
    # Optional clipping of large changes
    max_dq = args.max_dq
    mask = np.abs(dq) > max_dq
    if mask.any():
        dq[mask] = np.sign(dq[mask]) * max_dq

    qvel = np.vstack([dq / dt, np.zeros((1, dq.shape[1]))])
    # Build states and next_states
    states = np.hstack([qpos_arr[:-1], qvel[:-1]])
    next_states = np.hstack([qpos_arr[1:], qvel[1:]])

    # Actions = dq
    actions = dq
    # Normalize via env.action_space.high
    try:
        env = make_g1_env()
        high = env.action_space.high
        env.close()
        scale = np.where(np.abs(high) > 1e-6, high, 1.0)
        actions = np.clip(actions / scale, -args.clip, args.clip)
    except Exception:
        print("Warning: Could not normalize actions via env; skipping")

    # Rewards and dones
    n = states.shape[0]
    rewards = np.zeros((n, 1), dtype=np.float32)
    dones = np.zeros((n, 1), dtype=bool)
    dones[-1] = True

    # Save buffer
    buffer = {
        'state': torch.tensor(states, dtype=torch.float32),
        'action': torch.tensor(actions, dtype=torch.float32),
        'reward': torch.tensor(rewards),
        'done': torch.tensor(dones),
        'next_state': torch.tensor(next_states, dtype=torch.float32)
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(buffer, args.out)
    print(f"Hybrid expert buffer saved to {args.out} with {n} transitions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create hybrid expert buffer for GAIL+PPO")
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='buffers/expert_hybrid.pth')
    parser.add_argument('--model_xml', default='g1_23dof_simplified.xml')
    parser.add_argument('--time_col', default='Timestamp')
    parser.add_argument('--lowpass', type=float, default=6.0)
    parser.add_argument('--fs', type=float, default=200.0)
    parser.add_argument('--savgol_window', type=int, default=9)
    parser.add_argument('--savgol_order', type=int, default=3)
    parser.add_argument('--max_dq', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=0.9)
    args = parser.parse_args()
    create_hybrid_buffer(args)
