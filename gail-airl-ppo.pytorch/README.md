# GAIL and AIRL in PyTorch
This is a PyTorch implementation of Generative Adversarial Imitation Learning(GAIL)[[1]](#references) and Adversarial Inverse Reinforcement Learning(AIRL)[[2]](#references) based on PPO[[3]](#references). I tried to make it easy for readers to understand the algorithm. Please let me know if you have any questions.

## Setup
You can set up the environment using the following steps:

1. Create a Python 3.10 virtual environment:
   ```bash
   python3.10 -m venv mujoco-env-py310-compat
   source mujoco-env-py310-compat/bin/activate
   ```

2. Install required packages:
   ```bash
   pip install numpy==1.24.3 torch gymnasium tqdm tensorboard
   ```

3. Install MuJoCo:
   ```bash
   mkdir -p ~/.mujoco
   curl -OL https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz
   tar -xf mujoco210-macos-x86_64.tar.gz -C ~/.mujoco/
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
   export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210/
   pip install mujoco
   pip install "gymnasium[mujoco]"
   ```

## Example

### Train expert
You can train experts using Soft Actor-Critic(SAC)[[4,5]](#references).

```bash
python train_expert.py --env_id InvertedPendulum-v2 --num_steps 10000 --seed 0
```

### Collect demonstrations
You need to collect demonstrations using trained expert's weight. Note that `--std` specifies the standard deviation of the gaussian noise added to the action, and `--p_rand` specifies the probability the expert acts randomly. We set `std` to 0.01 not to collect too similar trajectories.

```bash
python collect_demo.py \
    --env_id InvertedPendulum-v2 \
    --weight logs/InvertedPendulum-v2/sac/seed0-TIMESTAMP/model/step10000/actor.pth \
    --buffer_size 10000 --std 0.01 --p_rand 0.0 --seed 0
```

Replace TIMESTAMP with the actual timestamp from your training run.

### Train Imitation Learning
You can train IL using demonstrations:

```bash
python train_imitation.py \
    --algo gail --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size10000_std0.01_prand0.0.pth \
    --num_steps 10000 --eval_interval 1000 --rollout_length 2000 --seed 0
```

Try AIRL too:

```bash
python train_imitation.py \
    --algo airl --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size10000_std0.01_prand0.0.pth \
    --num_steps 10000 --eval_interval 1000 --rollout_length 2000 --seed 0
```

### Visualize Results
You can visualize the trained expert:

```bash
python visualize_policy.py --model_path logs/G1-v0/gail/seed0-20250423-0304/model/step6845000/actor.pth --num_episodes 3 --fps 5
```

View training metrics with TensorBoard:

```bash
tensorboard --logdir logs
```

## Performance
- SAC (Expert): ~12 points per episode
- GAIL: ~62 points per episode after 10,000 steps  
- AIRL: ~104 points per episode after 10,000 steps

## References
[[1]](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning) Ho, Jonathan, and Stefano Ermon. "Generative adversarial imitation learning." Advances in neural information processing systems. 2016.

[[2]](https://arxiv.org/abs/1710.11248) Fu, Justin, Katie Luo, and Sergey Levine. "Learning robust rewards with adversarial inverse reinforcement learning." arXiv preprint arXiv:1710.11248 (2017).

[[3]](https://arxiv.org/abs/1707.06347) Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).

[[4]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).

[[5]](https://arxiv.org/abs/1812.05905) Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
