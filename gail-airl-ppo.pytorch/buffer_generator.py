import os
import argparse
import pandas as pd
import numpy as np
import torch
from scipy.signal import butter, filtfilt, savgol_filter
import mujoco
from mujoco import MjModel, MjData, mj_step, mj_name2id, mjtObj
from g1_env import make_g1_env

# 1. Define mapping from mocap CSV columns to robot joint names
JOINT_MAPPING = {
    'LeftHip_pitch':    'left_hip_pitch_joint',
    'LeftHip_roll':     'left_hip_roll_joint',
    'LeftHip_yaw':      'left_hip_yaw_joint',
    'LeftKnee_flexion': 'left_knee_joint',
    'LeftAnkle_pitch':  'left_ankle_pitch_joint',
    'LeftAnkle_roll':   'left_ankle_roll_joint',
    'RightHip_pitch':   'right_hip_pitch_joint',
    'RightHip_roll':    'right_hip_roll_joint',
    'RightHip_yaw':     'right_hip_yaw_joint',
    'RightKnee_flexion':'right_knee_joint',
    'RightAnkle_pitch': 'right_ankle_pitch_joint',
    'RightAnkle_roll':  'right_ankle_roll_joint',
    'LeftShoulder_pitch':'left_shoulder_pitch_joint',
    'LeftShoulder_roll': 'left_shoulder_roll_joint',
    'LeftShoulder_yaw':  'left_shoulder_yaw_joint',
    'LeftElbow_flexion':'left_elbow_joint',
    'RightShoulder_pitch':'right_shoulder_pitch_joint',
    'RightShoulder_roll': 'right_shoulder_roll_joint',
    'RightShoulder_yaw':  'right_shoulder_yaw_joint',
    'RightElbow_flexion':'right_elbow_joint',
    'LeftWrist_pronation':'left_wrist_roll_joint',
    'RightWrist_pronation':'right_wrist_roll_joint',
    'Waist_yaw':          'waist_yaw_joint'
}

# 2. Filtering utilities

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# 3. Load & preprocess mocap: deg→rad + smoothing

def load_and_preprocess(csv_path, joint_columns, lowpass_cutoff, fs, savgol_window, savgol_order):
    df = pd.read_csv(csv_path)
    print(f"CSV file loaded. Columns: {list(df.columns)}")
    # Degrees → radians
    for col in joint_columns:
        df[col] = np.deg2rad(df[col].values)
    # Butterworth + Savitzky-Golay
    for col in joint_columns:
        df[col] = lowpass_filter(df[col].values, lowpass_cutoff, fs)
        if len(df[col]) > savgol_window:
            if savgol_window % 2 == 0:
                savgol_window += 1
            df[col] = savgol_filter(df[col].values, savgol_window, savgol_order)
    return df

# 4. Robust dt calc

def compute_dt(timestamps, default_dt):
    dt_vals = np.diff(timestamps)
    pos = dt_vals[dt_vals > 0]
    return max(pos.mean() if len(pos)>0 else default_dt, default_dt)

# 5. Clip to joint limits

def enforce_limits(angle, limits, joint_name='Unknown', scale_factor=1.0):
    lo, hi = limits
    # First, apply scaling to reduce the range of motion
    scaled_angle = angle * scale_factor
    # Then, clip to ensure within limits
    clipped_angle = np.clip(scaled_angle, lo, hi)
    if scaled_angle != angle:
        print(f"- Scaled {joint_name} angle from {angle:.4f} to {scaled_angle:.4f} (factor {scale_factor:.2f})")
    if clipped_angle != scaled_angle:
        print(f"- Clipped {joint_name} angle from {scaled_angle:.4f} to {clipped_angle:.4f} within limits [{lo:.4f}, {hi:.4f}]")
        if not hasattr(enforce_limits, 'clip_count'):
            enforce_limits.clip_count = {}
        enforce_limits.clip_count[joint_name] = enforce_limits.clip_count.get(joint_name, 0) + 1
    return clipped_angle

# Summary of clipping at the end of processing
def print_clipping_summary():
    if hasattr(enforce_limits, 'clip_count') and enforce_limits.clip_count:
        print("\n=== Clipping Summary ===")
        for joint, count in enforce_limits.clip_count.items():
            print(f"- {joint}: Clipped {count} times")
        print("=== End Clipping Summary ===\n")

# 6. Sanity checks

def sanity_checks(buffer_path, joint_limits, clip_val):
    buf = torch.load(buffer_path)
    states = buf['state'].numpy()
    actions = buf['action'].numpy()
    n_joints = len(joint_limits)
    print("\n=== Sanity Checks ===")
    # Qpos limits
    qpos = states[:, :n_joints]
    for i,(csv_c, jn) in enumerate(JOINT_MAPPING.items()):
        lo, hi = joint_limits[jn]
        vio = np.logical_or(qpos[:,i] < lo, qpos[:,i] > hi)
        if vio.any():
            print(f"- {jn}: {vio.sum()}/{qpos.shape[0]} out of limits (range [{lo:.4f}, {hi:.4f}])")
            # Log a few example values for debugging with deviation info
            vio_indices = np.where(vio)[0][:5]  # First 5 violations
            for idx in vio_indices:
                val = qpos[idx, i]
                deviation = min(abs(val - lo), abs(val - hi)) if val < lo or val > hi else 0
                direction = 'below lower limit' if val < lo else 'above upper limit'
                print(f"  - Frame {idx}: Value {val:.4f}, Deviation {deviation:.4f} {direction}")
    # Action range
    print(f"- Action range: min {actions.min():.4f}, max {actions.max():.4f} (clip {clip_val})")
    # State stats
    print(f"- State mean {states.mean():.4f}, std {states.std():.4f}")
    # Done flag
    dones = buf['done'].numpy().astype(bool)
    if dones.sum()==1 and dones[-1]:
        print("- Done flag OK")
    else:
        print(f"- Done flags {dones.sum()} (expected 1)")
    print("=== End Checks ===\n")

# 7. Main

def create_hybrid_buffer(args):
    joint_cols = list(JOINT_MAPPING.keys())
    df = load_and_preprocess(args.csv, joint_cols,
                              args.lowpass, args.fs,
                              args.savgol_window, args.savgol_order)
    # Load model & data
    model = MjModel.from_xml_path(args.model_xml)
    data = MjData(model)
    # Joint limits via name2id
    joint_limits = {}
    for jn in JOINT_MAPPING.values():
        jid = mj_name2id(model, mjtObj.mjOBJ_JOINT, jn)
        joint_limits[jn] = tuple(model.jnt_range[jid])

    # Retarget & simulate
    qpos_list = []
    timestamps = df[args.time_col].values if args.time_col in df else np.arange(len(df))/args.fs
    for frame_idx, (_, row) in enumerate(df.iterrows()):
        for csv_c, jn in JOINT_MAPPING.items():
            ang = enforce_limits(row[csv_c], joint_limits[jn], jn, args.scale_factor)
            jid = mj_name2id(model, mjtObj.mjOBJ_JOINT, jn)
            idx = model.jnt_qposadr[jid]
            data.qpos[idx] = ang
        mj_step(model, data)
        # Re-clip qpos after simulation to ensure within limits
        qpos_after_sim = data.qpos.copy()
        for jn in JOINT_MAPPING.values():
            jid = mj_name2id(model, mjtObj.mjOBJ_JOINT, jn)
            idx = model.jnt_qposadr[jid]
            lo, hi = joint_limits[jn]
            original_val = qpos_after_sim[idx]
            clipped_val = np.clip(original_val, lo, hi)
            if clipped_val != original_val:
                print(f"- Frame {frame_idx}: Post-sim clipped {jn} from {original_val:.4f} to {clipped_val:.4f} within limits [{lo:.4f}, {hi:.4f}]")
                if not hasattr(enforce_limits, 'post_sim_clip_count'):
                    enforce_limits.post_sim_clip_count = {}
                enforce_limits.post_sim_clip_count[jn] = enforce_limits.post_sim_clip_count.get(jn, 0) + 1
            qpos_after_sim[idx] = clipped_val
        qpos_list.append(qpos_after_sim)
        # Debug: Check a few frames to confirm clipping
        if frame_idx < 5 or frame_idx > len(df) - 6:
            for jn in JOINT_MAPPING.values():
                jid = mj_name2id(model, mjtObj.mjOBJ_JOINT, jn)
                idx = model.jnt_qposadr[jid]
                val = qpos_after_sim[idx]
                lo, hi = joint_limits[jn]
                if val < lo or val > hi:
                    print(f"- WARNING Frame {frame_idx}: {jn} value {val:.4f} still out of limits [{lo:.4f}, {hi:.4f}] after clipping")

    qpos_arr = np.stack(qpos_list)
    # Final check and clip before building buffer to ensure no out-of-limit values
    n_joints = len(JOINT_MAPPING)
    for frame_idx in range(qpos_arr.shape[0]):
        for i, jn in enumerate(JOINT_MAPPING.values()):
            lo, hi = joint_limits[jn]
            val = qpos_arr[frame_idx, i]
            if val < lo or val > hi:
                clipped_val = np.clip(val, lo, hi)
                print(f"- Final check Frame {frame_idx}: Clipped {jn} from {val:.4f} to {clipped_val:.4f} within limits [{lo:.4f}, {hi:.4f}]")
                if not hasattr(enforce_limits, 'final_clip_count'):
                    enforce_limits.final_clip_count = {}
                enforce_limits.final_clip_count[jn] = enforce_limits.final_clip_count.get(jn, 0) + 1
                qpos_arr[frame_idx, i] = clipped_val
        # Debug a few frames
        if frame_idx < 3 or frame_idx > qpos_arr.shape[0] - 4:
            for i, jn in enumerate(JOINT_MAPPING.values()):
                lo, hi = joint_limits[jn]
                val = qpos_arr[frame_idx, i]
                if val < lo or val > hi:
                    print(f"- WARNING Final check Frame {frame_idx}: {jn} value {val:.4f} still out of limits [{lo:.4f}, {hi:.4f}] after final clipping")

    # Summarize final clipping
    if hasattr(enforce_limits, 'final_clip_count') and enforce_limits.final_clip_count:
        print("\n=== Final Clipping Summary ===")
        for joint, count in enforce_limits.final_clip_count.items():
            print(f"- {joint}: Clipped {count} times before buffer creation")
        print("=== End Final Clipping Summary ===\n")

    dt = compute_dt(timestamps, 1/args.fs)
    dq = np.diff(qpos_arr, axis=0)
    mask = np.abs(dq) > args.max_dq
    dq[mask] = np.sign(dq[mask]) * args.max_dq
    qvel = np.vstack([dq/ dt, np.zeros((1, dq.shape[1]))])

    states = np.hstack([qpos_arr[:-1], qvel[:-1]])
    next_states = np.hstack([qpos_arr[1:], qvel[1:]])
    actions = dq.copy()
    # Normalize actions
    try:
        env = make_g1_env()
        high = env.action_space.high
        env.close()
        scale = np.where(np.abs(high)>1e-6, high, 1.0)
        actions = np.clip(actions/ scale, -args.clip, args.clip)
    except Exception:
        print("Warning: action normalization skipped.")

    # Save
    n = states.shape[0]
    buffer = {'state': torch.tensor(states),
              'action': torch.tensor(actions),
              'reward': torch.zeros((n,1)),
              'done': torch.tensor(np.vstack([np.zeros((n-1,1)), [[1]]]), dtype=torch.bool),
              'next_state': torch.tensor(next_states)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(buffer, args.out)
    print(f"Hybrid buffer saved {args.out} ({n} transitions)")

    if args.sanity:
        sanity_checks(args.out, joint_limits, args.clip)

    print_clipping_summary()
    # Summary of post-simulation clipping
    if hasattr(enforce_limits, 'post_sim_clip_count') and enforce_limits.post_sim_clip_count:
        print("\n=== Post-Simulation Clipping Summary ===")
        for joint, count in enforce_limits.post_sim_clip_count.items():
            print(f"- {joint}: Clipped {count} times after simulation")
        print("=== End Post-Simulation Clipping Summary ===\n")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='buffers/expert_hybrid.pth')
    p.add_argument('--model_xml', default='g1_23dof_simplified.xml')
    p.add_argument('--time_col', default='Timestamp')
    p.add_argument('--lowpass', type=float, default=6.0)
    p.add_argument('--fs', type=float, default=200.0)
    p.add_argument('--savgol_window', type=int, default=9)
    p.add_argument('--savgol_order', type=int, default=3)
    p.add_argument('--max_dq', type=float, default=0.1)
    p.add_argument('--clip', type=float, default=0.9)
    p.add_argument('--sanity', action='store_true')
    p.add_argument('--scale_factor', type=float, default=0.85, help='Scaling factor for joint angles to reduce range of motion (0-1)')
    args = p.parse_args()
    create_hybrid_buffer(args)
