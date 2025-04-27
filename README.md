# CS558_final_project_robot_learning



## Scripts

### `make_buffer.py`
- **Purpose**: Convert raw CSV motion data into an imitation‐learning buffer (`.pth`) of (state, action, reward, done, next_state) tensors.
- **Key steps**:
  1. Reads CSV, excludes timestamp and any extra columns.  
  2. Computes finite‐difference velocities and pads.  
  3. Builds dummy rewards/dones and (state, action, next_state) pairs.  
  4. Optionally normalizes actions by the env’s action bounds.  
  5. Saves output as `--out` (.pth file).
- **Example**:  
  ```bash
  python make_buffer.py \
    --csv data/your_demonstrations.csv \
    --out buffers/expert.pth \
    --time_col Timestamp \
    --exclude extra_col1 extra_col2
  ```

### `train_imitation_stable.py`
- **Purpose**: Train a GAIL or AIRL policy via PPO, using the expert buffer.
- **Key steps**:
  1. Loads expert buffer (`SerializedBuffer`).  
  2. Spawns train & test environments.  
  3. Builds chosen algo (`--algo gail|airl`) with PPO inner loop.  
  4. Runs updates, logs to `logs/<env_id>/<algo>/seed…`.  
  5. Emergency-saves on error.
- **Example**:  
  ```bash
  python train_imitation_stable.py \
    --buffer buffers/expert.pth \
    --env_id G1-v0 \
    --algo gail \
    --cuda \
    --seed 0 \
    --lr 1e-4 \
    --batch_size 64 \
    --max_grad_norm 1.0 \
    --entropy_coef 0.01 \
    --gamma 0.995 \
    --lambd 0.97 \
    --clip_eps 0.2 \
    --epoch_ppo 10 \
    --rollout_length 2048 \
    --num_steps 1000000 \
    --eval_interval 5000
  ```

### `evaluate_policy.py`
- **Purpose**: Headless evaluation of a trained policy, printing per-step and aggregate metrics.
- **Key steps**:
  1. Creates fresh G1 environment.  
  2. Loads `actor.pth` from `--model_dir`.  
  3. Runs N episodes (default 5), logging reward, success rate.
- **Example**:  
  ```bash
  python evaluate_policy.py \
    --model_dir logs/G1-v0/gail/seed0-20230601-1530 \
    --episodes 10 \
    --seed 0
  ```

### `visualize_policy.py`
- **Purpose**: Interactive MuJoCo viewer of rollouts to qualitatively inspect behavior.
- **Key steps**:
  1. Loads `actor.pth` via `--model_path`.  
  2. Attaches `mujoco_viewer` to the G1Env.  
  3. Runs episodes, rendering at your chosen FPS.
- **Example**:  
  ```bash
  python visualize_policy.py \
    --model_path logs/G1-v0/gail/seed0-20230601-1530/actor.pth \
    --num_episodes 3 \
    --max_steps 1000 \
    --fps 30 \
    --seed 0 \
    --cuda
  ```

## Full Test Run

1. **Generate expert buffer**  
   ```bash
   python make_buffer.py \
     --csv data/your_demonstrations.csv \
     --out buffers/expert.pth
   ```

2. **Train imitation policy**  
   ```bash
   python train_imitation_stable.py \
     --buffer buffers/expert.pth \
     --env_id G1-v0 \
     --algo gail \
     --cuda \
     --seed 0
   ```

3. **Evaluate performance**  
   ```bash
   python evaluate_policy.py \
     --model_dir logs/G1-v0/gail/seed0-*/ \
     --episodes 5
   ```

4. **Visualize rollouts**  
   ```bash
   python visualize_policy.py \
     --model_path logs/G1-v0/gail/seed0-*/actor.pth \
     --num_episodes 3 \
     --fps 30
   ```

Replace wildcards (`*`) with the actual timestamped folder names produced during training.