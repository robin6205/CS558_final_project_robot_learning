import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import our custom G1 environment instead of the standard environments
from g1_env import make_g1_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class BehaviorCloning:
    """Simple Behavior Cloning implementation for pre-training a policy."""
    
    def __init__(self, algo, device, batch_size=64):
        self.algo = algo
        self.device = device
        self.batch_size = batch_size
        
        # Use actor from the algorithm
        self.actor = self.algo.actor
        
        # Create a separate optimizer with a higher learning rate for BC
        self.optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        print(f"Behavior Cloning initialized with batch_size={batch_size}")
        
    def update(self, states, actions, iterations=1000):
        """Update the policy using behavior cloning."""
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        
        n_samples = len(states)
        indexes = np.arange(n_samples)
        
        loss_history = []
        best_loss = float('inf')
        best_state = None
        patience_counter = 0
        patience = 20  # Stop if no improvement for 20 iterations
        
        pbar = tqdm(range(iterations), desc="BC Pre-training")
        for i in pbar:
            # Sample a mini-batch
            np.random.shuffle(indexes)
            batch_indexes = indexes[:self.batch_size]
            
            batch_states = states[batch_indexes]
            batch_actions = actions[batch_indexes]
            
            # Zero gradients
            self.optim.zero_grad()
            
            # Forward pass
            pred_actions = self.actor(batch_states)
            
            # Calculate MSE loss
            loss = nn.MSELoss()(pred_actions, batch_actions)
            
            # Backward pass and optimize
            loss.backward()
            
            # Clip gradients for stability
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            
            self.optim.step()
            
            loss_history.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self.actor.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping if the loss is very low or no improvement
            if (i > 500 and loss.item() < 0.0005) or (patience_counter >= patience):
                print(f"Early stopping at iteration {i} with loss {loss.item():.6f}")
                break
        
        # Restore best model
        if best_state is not None:
            self.actor.load_state_dict(best_state)
            
        print(f"Final BC loss: {loss_history[-1]:.6f}, Best loss: {best_loss:.6f}")
        return loss_history
    
    def augment_data(self, states, actions):
        """Create mirrored versions of the data to increase the training set.
        For the G1 robot, we mirror left-right joints."""
        print("Augmenting expert data with mirroring...")
        
        # Create deep copies to avoid modifying originals
        mirrored_states = states.copy()
        mirrored_actions = actions.copy()
        
        # Define joint indices to swap (left to right and vice versa)
        # Based on the 23 DOF G1 robot:
        # Note: These indices need to be verified for the specific model
        swap_indices = [
            # Hip joints (swap left and right)
            (0, 3),  # LeftHip_pitch <-> RightHip_pitch
            (1, 4),  # LeftHip_roll <-> RightHip_roll (need to negate)
            (2, 5),  # LeftHip_yaw <-> RightHip_yaw (need to negate)
            
            # Knee joints
            (6, 7),  # LeftKnee_flexion <-> RightKnee_flexion
            
            # Ankle joints 
            (8, 10),  # LeftAnkle_pitch <-> RightAnkle_pitch
            (9, 11),  # LeftAnkle_roll <-> RightAnkle_roll (need to negate)
            
            # Shoulder joints
            (12, 15),  # LeftShoulder_pitch <-> RightShoulder_pitch
            (13, 16),  # LeftShoulder_roll <-> RightShoulder_roll (need to negate)
            (14, 17),  # LeftShoulder_yaw <-> RightShoulder_yaw (need to negate)
            
            # Elbow joints
            (18, 19),  # LeftElbow_flexion <-> RightElbow_flexion
            
            # Wrist joints
            (20, 21),  # LeftWrist_pronation <-> RightWrist_pronation (need to negate)
        ]
        
        # Joints that should be negated when mirrored
        negate_indices = [1, 4, 2, 5, 9, 11, 13, 16, 14, 17, 20, 21]
        
        # Swap joint positions in states
        half_dim = states.shape[1] // 2  # Half of state is position, half is velocity
        
        # Process positions first (first half of state vector)
        for i, j in swap_indices:
            # Swap positions
            tmp = mirrored_states[:, i].copy()
            mirrored_states[:, i] = mirrored_states[:, j]
            mirrored_states[:, j] = tmp
            
            # Swap velocities (offset by half_dim)
            tmp = mirrored_states[:, i + half_dim].copy()
            mirrored_states[:, i + half_dim] = mirrored_states[:, j + half_dim]
            mirrored_states[:, j + half_dim] = tmp
        
        # Negate the appropriate indices for both positions and velocities
        for i in negate_indices:
            mirrored_states[:, i] *= -1  # Negate positions
            mirrored_states[:, i + half_dim] *= -1  # Negate velocities
            
        # Apply the same swapping and negation for actions
        for i, j in swap_indices:
            # Swap actions
            tmp = mirrored_actions[:, i].copy()
            mirrored_actions[:, i] = mirrored_actions[:, j]
            mirrored_actions[:, j] = tmp
            
        # Negate appropriate actions
        for i in negate_indices:
            mirrored_actions[:, i] *= -1
        
        # We also need to negate the x-axis motion to get proper mirroring
        # This depends on the specific robot model - for G1 walking right,
        # we flip to make it walk left by negating goal-related values
        
        # Combine original and mirrored data
        aug_states = np.vstack([states, mirrored_states])
        aug_actions = np.vstack([actions, mirrored_actions])
        
        print(f"Augmented data: {len(states)} original samples + {len(mirrored_states)} mirrored = {len(aug_states)} total")
        return aug_states, aug_actions

def run(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create the environment - use our custom G1 environment
    if args.env_id == 'G1Raw-v0':
        print("Using custom G1 environment")
        print("Environment features:")
        print("- 3m goal distance for side-stepping")
        print("- Strong upright orientation reward")
        print("- Torso-based goal distance measurement")
        env = make_g1_env()
        env_test = make_g1_env()
        
        # Set command values for velocity tracking
        if hasattr(env.unwrapped, 'vx_ref'):
            # Set a small reference velocity in the x direction (to the right)
            env.unwrapped.vx_ref = 0.5  # 0.5 m/s to the right (x-axis) for tracking
            print(f"Setting reference velocity command: vx={env.unwrapped.vx_ref} m/s")
            
            # Set the same for evaluation environment
            env_test.unwrapped.vx_ref = 0.5
        
        # Adjust reward weights for better stability
        if hasattr(env.unwrapped, 'lin_vel_tracking_weight'):
            # Increase linear velocity tracking weight
            env.unwrapped.lin_vel_tracking_weight = 2.0
            env_test.unwrapped.lin_vel_tracking_weight = 2.0
            
            # Increase roll/pitch stability weight
            env.unwrapped.roll_pitch_penalty_weight = 2.5
            env_test.unwrapped.roll_pitch_penalty_weight = 2.5
            
            print(f"Adjusted reward weights for better stability")
    else:
        # Fallback to standard Gym environments
        from gail_airl_ppo.env import make_env
        env = make_env(args.env_id)
        env_test = make_env(args.env_id)
    
    # Load expert buffer
    device = torch.device("cuda" if args.cuda else "cpu")
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=device
    )
    
    # Print debugging information
    print(f"Environment observation space shape: {env.observation_space.shape}")
    print(f"Environment action space shape: {env.action_space.shape}")
    print(f"Expert buffer state shape: {buffer_exp.states.shape}")
    print(f"Expert buffer action shape: {buffer_exp.actions.shape}")
    print(f"Using state history length: {args.history_length}")
    
    # Check for and fix any NaN or Inf values in the expert buffer
    def check_and_fix_tensor(tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: Found NaN or Inf in expert buffer {name}. Replacing with zeros.")
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor
    
    buffer_exp.states = check_and_fix_tensor(buffer_exp.states, "states")
    buffer_exp.actions = check_and_fix_tensor(buffer_exp.actions, "actions")
    buffer_exp.rewards = check_and_fix_tensor(buffer_exp.rewards, "rewards")
    buffer_exp.dones = check_and_fix_tensor(buffer_exp.dones, "dones")
    buffer_exp.next_states = check_and_fix_tensor(buffer_exp.next_states, "next_states")
    
    # Create algorithm with modified hyperparameters for stability
    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        seed=args.seed,
        rollout_length=args.rollout_length,
        # Specific stability improvements
        lr_actor=args.lr,
        lr_critic=args.lr,
        lr_disc=args.lr_disc,
        epoch_ppo=args.epoch_ppo,
        max_grad_norm=args.max_grad_norm,
        coef_ent=args.entropy_coef,
        # Keep other parameters
        gamma=args.gamma,
        lambd=args.lambd,
        clip_eps=args.clip_eps,
        batch_size=args.batch_size
    )

    # Modify algo's actor to use state history if enabled
    if args.history_length > 1:
        from StateIndependentGaussianPolicy import StateIndependentGaussianPolicy
        
        # Get the original actor's device
        actor_device = next(algo.actor.parameters()).device
        
        # Create new actor with history support
        new_actor = StateIndependentGaussianPolicy(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            hidden_units=(64, 64),  # Match original
            hidden_activation=torch.nn.Tanh(),  # Match original
            history_length=args.history_length
        ).to(actor_device)
        
        # Initialize with weights from original actor where possible
        # This transfers weights that match in shape
        matching_state_dict = {}
        original_state_dict = algo.actor.state_dict()
        
        for name, param in new_actor.state_dict().items():
            # Only copy parameters that have matching shapes
            if name in original_state_dict and original_state_dict[name].shape == param.shape:
                matching_state_dict[name] = original_state_dict[name]
        
        # Load the matching parameters
        if matching_state_dict:
            new_actor.load_state_dict(matching_state_dict, strict=False)
            print(f"Transferred {len(matching_state_dict)}/{len(new_actor.state_dict())} parameters from original actor")
        
        # Replace the actor in the algorithm
        original_actor = algo.actor
        algo.actor = new_actor
        print(f"Replaced standard actor with history-aware actor (history_length={args.history_length})")

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}'
    )
    os.makedirs(log_dir, exist_ok=True)
    
    # Behavior cloning pre-training if enabled
    if args.bc_iterations > 0:
        print(f"Starting behavior cloning pre-training for {args.bc_iterations} iterations")
        
        # Create BC trainer using the algo's actor
        bc_trainer = BehaviorCloning(
            algo=algo,
            device=device,
            batch_size=args.batch_size
        )
        
        # Extract states and actions for training
        expert_states = buffer_exp.states.cpu().numpy()
        expert_actions = buffer_exp.actions.cpu().numpy()
        
        # Optionally augment the expert data with mirroring
        if args.augment_data:
            expert_states, expert_actions = bc_trainer.augment_data(expert_states, expert_actions)
        
        # Apply noise augmentation if enabled
        if args.action_noise > 0:
            print(f"Adding Gaussian noise (σ={args.action_noise}) to actions for robustness")
            noise = np.random.normal(0, args.action_noise, expert_actions.shape)
            expert_actions_noisy = expert_actions + noise
            # Clip actions to valid range
            expert_actions_noisy = np.clip(expert_actions_noisy, -1.0, 1.0)
        else:
            expert_actions_noisy = expert_actions
        
        # Pre-train the actor with behavior cloning
        bc_loss_history = bc_trainer.update(
            states=expert_states,
            actions=expert_actions_noisy,
            iterations=args.bc_iterations
        )
        
        # Save pre-trained model
        algo.save_models(os.path.join(log_dir, 'bc_pretrained'))
        print(f"Saved BC pre-trained model to {os.path.join(log_dir, 'bc_pretrained')}")

    # Define a simple standalone trainer instead of extending the existing Trainer
    class SimpleTrainer:
        def __init__(self, env, env_test, algo, log_dir, num_steps, eval_interval, seed, history_length=1):
            self.env = env
            self.env_test = env_test
            self.algo = algo
            self.log_dir = log_dir
            self.num_steps = num_steps
            self.eval_interval = eval_interval
            self.seed = seed
            self.history_length = history_length
            
            # Set up logger
            self.writer = SummaryWriter(log_dir=log_dir)
            
            self.steps = 0
            self.episodes = 0
            self.eval_episodes = 0
            self.best_reward = -float('inf')
        
        def reset_history(self):
            """Reset the state history buffer"""
            if hasattr(self.algo.actor, 'state_history') and self.algo.actor.state_history is not None:
                # Use the actor's built-in history
                self.algo.actor.state_history = torch.zeros_like(self.algo.actor.state_history)
        
        def evaluate(self):
            """Evaluate the current policy"""
            print("\nEvaluating policy...")
            eval_rewards = []
            eval_steps = []
            
            # Run evaluation episodes
            for _ in range(3):  # Use 3 episodes for faster evaluation
                state, _ = self.env_test.reset()
                done = False
                episode_reward = 0
                episode_steps = 0
                
                # Reset history for evaluation
                self.reset_history()
                
                while not done:
                    action = self.algo.exploit(state)
                    next_state, reward, terminated, truncated, info = self.env_test.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                
                eval_rewards.append(episode_reward)
                eval_steps.append(episode_steps)
            
            # Calculate statistics
            mean_reward = sum(eval_rewards) / len(eval_rewards)
            mean_steps = sum(eval_steps) / len(eval_steps)
            
            # Log results
            self.eval_episodes += 1
            print(f"Evaluation {self.eval_episodes}: Mean reward = {mean_reward:.2f}, Mean steps = {mean_steps:.2f}")
            self.writer.add_scalar('eval/mean_reward', mean_reward, self.steps)
            self.writer.add_scalar('eval/mean_steps', mean_steps, self.steps)
            
            # Save best model
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                print(f"New best reward: {mean_reward:.2f}, saving model...")
                self.algo.save_models(os.path.join(self.log_dir, 'best'))
        
        def train(self):
            """Train the agent"""
            print("Starting training...")
            state, _ = self.env.reset()
            self.reset_history()
            
            # Training loop
            while self.steps < self.num_steps:
                # Collect experience
                for _ in range(min(self.algo.rollout_length, self.num_steps - self.steps)):
                    # Get action
                    action, log_pi = self.algo.explore(state)
                    
                    # Step environment
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    # Store transition
                    self.algo.buffer.append(state, action, reward, done, log_pi, next_state)
                    
                    # Update state
                    state = next_state
                    self.steps += 1
                    
                    # Log info
                    if self.steps % 100 == 0:
                        print(f"Steps: {self.steps}/{self.num_steps}")
                    
                    # Reset on episode end
                    if done:
                        self.episodes += 1
                        state, _ = self.env.reset()
                        self.reset_history()
                        print(f"Episode {self.episodes} completed")
                    
                    # Evaluate periodically
                    if self.eval_interval > 0 and self.steps % self.eval_interval == 0:
                        self.evaluate()
                        # Save checkpoint
                        self.algo.save_models(os.path.join(self.log_dir, f'step_{self.steps}'))
                
                # Update policy with collected experience
                update_info = self.algo.update(self.writer)
                
                # Log update info if available
                if update_info:
                    for key, value in update_info.items():
                        self.writer.add_scalar(f'update/{key}', value, self.steps)
            
            # Final evaluation and saving
            self.evaluate()
            self.algo.save_models(os.path.join(self.log_dir, 'final'))
            print("Training completed!")

    # Create trainer with custom extension if history is enabled
    if args.history_length > 1:
        trainer = SimpleTrainer(
            env=env,
            env_test=env_test,
            algo=algo,
            log_dir=log_dir,
            num_steps=args.num_steps,
            eval_interval=args.eval_interval,
            seed=args.seed,
            history_length=args.history_length
        )
    else:
        # Use standard trainer for history_length=1
        trainer = SimpleTrainer(
            env=env,
            env_test=env_test,
            algo=algo,
            log_dir=log_dir,
            num_steps=args.num_steps,
            eval_interval=args.eval_interval,
            seed=args.seed,
            history_length=args.history_length
        )
    
    print(f"Starting training with algorithm: {args.algo}")
    print(f"Rollout length: {args.rollout_length}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}, Discriminator LR: {args.lr_disc}, Max grad norm: {args.max_grad_norm}")
    print(f"Entropy coefficient: {args.entropy_coef}, Gamma: {args.gamma}")
    print(f"Lambda: {args.lambd}, Clip epsilon: {args.clip_eps}")
    print(f"PPO epochs: {args.epoch_ppo}")
    print(f"State history length: {args.history_length}")
    print(f"Logs will be saved to: {log_dir}")
    
    # Start training
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        # Save model on error to preserve progress
        algo.save_models(os.path.join(log_dir, f'emergency_save'))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True, help='Path to expert buffer')
    p.add_argument('--rollout_length', type=int, default=1024, help='Rollout length')
    p.add_argument('--num_steps', type=int, default=1000000, help='Number of training steps')
    p.add_argument('--eval_interval', type=int, default=5000, help='Evaluation interval')
    p.add_argument('--env_id', type=str, default='G1Raw-v0', help='Environment ID')
    p.add_argument('--algo', type=str, default='gail', help='Algorithm (gail or airl)')
    p.add_argument('--cuda', action='store_true', help='Use CUDA')
    p.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Additional stability parameters
    p.add_argument('--lr', type=float, default=2e-5, help='Learning rate for actor and critic')
    p.add_argument('--lr_disc', type=float, default=5e-6, help='Learning rate for discriminator')
    p.add_argument('--batch_size', type=int, default=64, help='Batch size')
    p.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')
    p.add_argument('--entropy_coef', type=float, default=0.0005, help='Entropy coefficient')
    p.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    p.add_argument('--lambd', type=float, default=0.95, help='GAE lambda')
    p.add_argument('--clip_eps', type=float, default=0.1, help='PPO clip epsilon')
    p.add_argument('--epoch_ppo', type=int, default=10, help='PPO epochs per update')
    
    # Behavior cloning pre-training parameters
    p.add_argument('--bc_iterations', type=int, default=12000, 
                  help='Number of BC iterations (0 to disable)')
    p.add_argument('--augment_data', action='store_true', 
                  help='Augment expert data with mirroring')
    p.add_argument('--action_noise', type=float, default=0.01, 
                  help='Add Gaussian noise to actions during BC training for robustness')
    
    # State history parameter
    p.add_argument('--history_length', type=int, default=1,
                  help='Number of previous states to include in observation')
    
    args = p.parse_args()
    run(args) 