import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from g1_env import make_g1_env
from bc_pretrain import build_mlp
import os
import logging
import random
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GAILPPO")

def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

# 1. Actor–Critic with shared backbone
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, learn_std=True, log_std_min=-20, log_std_max=2):
        super().__init__()
        # Use ReLU to prevent exploding gradients
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[-1]),
            nn.ReLU()
        )
        
        # Use smaller initializations for the policy head
        self.pi_mu = nn.Sequential(
            nn.Linear(hidden_sizes[-1], act_dim),
            nn.Tanh()  # Bound outputs to [-1, 1]
        )
        
        # Learned log standard deviation
        self.learn_std = learn_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        if learn_std:
            # Initialize log_std with values that give reasonable exploration
            self.log_std = nn.Parameter(torch.ones(act_dim) * -1.0)
        
        # Value function head
        self.v = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with appropriate gain for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use sqrt(2) gain which is standard for ReLU networks
                nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        # Apply input range check and normalization
        if torch.isnan(obs).any():
            # Print details about NaN values
            nan_indices = torch.nonzero(torch.isnan(obs))
            logger.warning(f"NaN values in observations at indices: {nan_indices[:10]}... (showing first 10)")
            logger.warning(f"Total NaN values: {torch.isnan(obs).sum().item()}/{obs.numel()}")
            
            # Replace NaNs with zeros
            obs = torch.nan_to_num(obs, nan=0.0)
        
        # Forward pass through shared layers
        h = self.shared(obs)
        
        # Get policy and value outputs
        pi_mu = self.pi_mu(h)
        v_out = self.v(h).squeeze(-1)
        
        # Get standard deviation
        if self.learn_std:
            # Clamp log_std to prevent extreme exploration behavior
            log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
            # Convert to std_dev (always positive) using exp
            std_dev = torch.exp(log_std)
        else:
            std_dev = torch.ones_like(pi_mu) * 0.1
        
        # Final NaN check and replacement
        if torch.isnan(pi_mu).any():
            nan_indices = torch.nonzero(torch.isnan(pi_mu))
            logger.warning(f"NaN values in pi_mu at indices: {nan_indices[:10]}... (showing first 10)")
            logger.warning(f"Total NaN values: {torch.isnan(pi_mu).sum().item()}/{pi_mu.numel()}")
            pi_mu = torch.nan_to_num(pi_mu, nan=0.0)
            
        if torch.isnan(v_out).any():
            nan_indices = torch.nonzero(torch.isnan(v_out))
            logger.warning(f"NaN values in v_out at indices: {nan_indices[:10]}... (showing first 10)")
            logger.warning(f"Total NaN values: {torch.isnan(v_out).sum().item()}/{v_out.numel()}")
            v_out = torch.nan_to_num(v_out, nan=0.0)
            
        return pi_mu, std_dev, v_out

# 2. Discriminator for GAIL
class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, 1, hidden_sizes)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)

# 3. GAIL + PPO Trainer
class GAILPPO:
    def __init__(self, env, expert_buffer, init_policy_path, args):
        self.env = env
        self.args = args
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        # Set device
        self.device = args.device
        
        # Set seed if provided
        if args.seed is not None:
            set_seed(args.seed)
            try:
                env.seed(args.seed)
            except:
                logger.warning(f"Environment doesn't support seeding or seed {args.seed} failed")
        
        # Load expert data
        logger.info(f"Loading expert data from {expert_buffer}")
        data = torch.load(expert_buffer)
        # Ensure expert data is float32
        self.expert_obs = data['state'][:, :obs_dim].to(dtype=torch.float32)
        self.expert_act = data['action'].to(dtype=torch.float32)
        logger.info(f"Loaded {len(self.expert_obs)} expert demonstrations")
        
        # Create persistent expert dataset for faster loading
        # We'll only add policy data during training
        self.expert_ds = TensorDataset(
            self.expert_obs.to(self.device), 
            self.expert_act.to(self.device),
            torch.ones(len(self.expert_obs), device=self.device)  # Expert labels = 1
        )
        self.expert_loader = DataLoader(
            self.expert_ds, 
            batch_size=min(args.batch_size, len(self.expert_ds)), 
            shuffle=True,
            drop_last=False
        )
        
        # Models
        self.ac = ActorCritic(
            obs_dim, act_dim, args.hidden_sizes, 
            learn_std=args.learn_std,
            log_std_min=args.log_std_min,
            log_std_max=args.log_std_max
        ).to(args.device)
        
        # Load DAgger policy and transfer to AC model
        if init_policy_path:
            logger.info(f"Loading DAgger policy from {init_policy_path}")
            try:
                # Check if file exists
                if not os.path.exists(init_policy_path):
                    logger.warning(f"Policy file {init_policy_path} does not exist. Using randomly initialized model.")
                else:
                    # Load the DAgger policy
                    dagger_state_dict = torch.load(init_policy_path, map_location=args.device)
                    
                    try:
                        # Try to directly load shared component if module names match
                        if 'shared.0.weight' in dagger_state_dict:
                            # Modern keys with shared. prefix
                            shared_dict = {k: v for k, v in dagger_state_dict.items() if k.startswith('shared.')}
                            if shared_dict:
                                self.ac.shared.load_state_dict(shared_dict)
                                logger.info(f"Successfully loaded shared weights from DAgger model using direct loading")
                        else:
                            # Fall back to manual key-by-key mapping
                            shared_keys = []
                            for key in dagger_state_dict.keys():
                                # Map DAgger keys to AC keys - standard MLP architecture from bc_pretrain
                                if key == '0.weight':
                                    if dagger_state_dict[key].shape == self.ac.shared[0].weight.shape:
                                        self.ac.shared[0].weight.data.copy_(dagger_state_dict[key])
                                        shared_keys.append(key)
                                elif key == '0.bias':
                                    if dagger_state_dict[key].shape == self.ac.shared[0].bias.shape:
                                        self.ac.shared[0].bias.data.copy_(dagger_state_dict[key])
                                        shared_keys.append(key)
                                elif key == '2.weight':
                                    if dagger_state_dict[key].shape == self.ac.shared[2].weight.shape:
                                        self.ac.shared[2].weight.data.copy_(dagger_state_dict[key])
                                        shared_keys.append(key)
                                elif key == '2.bias':
                                    if dagger_state_dict[key].shape == self.ac.shared[2].bias.shape:
                                        self.ac.shared[2].bias.data.copy_(dagger_state_dict[key])
                                        shared_keys.append(key)
                                elif key == '4.weight':
                                    if dagger_state_dict[key].shape == self.ac.pi_mu[0].weight.shape:
                                        self.ac.pi_mu[0].weight.data.copy_(dagger_state_dict[key])
                                        shared_keys.append(key)
                                elif key == '4.bias':
                                    if dagger_state_dict[key].shape == self.ac.pi_mu[0].bias.shape:
                                        self.ac.pi_mu[0].bias.data.copy_(dagger_state_dict[key])
                                        shared_keys.append(key)
                            
                            if shared_keys:
                                logger.info(f"Successfully loaded weights from DAgger model: {shared_keys}")
                            else:
                                logger.warning("No compatible weights found between DAgger and ActorCritic models")
                    except Exception as e:
                        logger.error(f"Error transferring weights: {e}")
                
            except Exception as e:
                logger.error(f"Error loading DAgger policy: {e}")
                logger.info("Continuing with randomly initialized ActorCritic model")
        else:
            logger.info("No initial policy provided. Using randomly initialized ActorCritic model.")
        
        # Discriminator for GAIL
        self.disc = Discriminator(obs_dim, act_dim, args.hidden_sizes).to(args.device)
        
        # Optimizers
        self.ac_opt = optim.Adam(self.ac.parameters(), lr=args.lr, weight_decay=args.ac_wd)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=args.disc_lr, weight_decay=args.disc_wd)
        
        # Hyperparameters
        self.gamma = args.gamma
        self.lam = args.lam
        self.clip = args.clip
        self.env_coef = args.env_coef

    def collect_trajectories(self, batch_size, max_steps_per_episode=1000):
        """Collect trajectories, ensuring complete episodes and tracking episode boundaries."""
        obs_buf, act_buf, ret_buf, val_buf, logp_buf, done_buf, ep_start_buf = [], [], [], [], [], [], []
        
        # Initialize tracking variables
        o, _ = self.env.reset()
        episode_steps = 0
        episode_count = 0
        total_steps = 0
        nan_found = False
        
        # Track episode boundaries for proper GAE calculation
        current_ep_start = 0
        
        # Use a separate model in eval mode for collecting trajectories
        with torch.no_grad():
            self.ac.eval()  # Set model to evaluation mode
            
            # Main collection loop
            while total_steps < batch_size:
                # Convert observation to tensor with proper type
                obs = torch.tensor(o, dtype=torch.float32, device=self.device)
                
                # Check observation for NaN values
                if torch.isnan(obs).any():
                    logger.warning(f"NaN in observation at ep {episode_count}, step {episode_steps}")
                    logger.warning(f"NaN count: {torch.isnan(obs).sum().item()}/{obs.numel()}")
                    obs = torch.nan_to_num(obs, nan=0.0)
                    nan_found = True
                
                # Get action prediction, standard deviation, and value
                pi_mu, std_dev, v = self.ac(obs)
                
                # Debug check for NaNs in network output
                if torch.isnan(pi_mu).any() or torch.isnan(v).any():
                    logger.warning(f"NaN in network output at ep {episode_count}, step {episode_steps}")
                    nan_found = True
                    
                    # Replace NaNs with zeros
                    pi_mu = torch.nan_to_num(pi_mu, nan=0.0)
                    v = torch.nan_to_num(v, nan=0.0)
                
                # Create action distribution
                dist = torch.distributions.Normal(pi_mu, std_dev)
                
                # Sample action and get log probability
                action = dist.rsample()
                logp = dist.log_prob(action).sum()
                
                # Clamp actions to be within environment limits
                action_np = action.detach().cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)  # Most environments use [-1, 1]
                
                # Step in environment
                try:
                    o2, r_env, terminated, truncated, info = self.env.step(action_np)
                    done = terminated or truncated
                except Exception as e:
                    logger.warning(f"Exception during env.step: {e}")
                    # Reset the environment and move to next iteration
                    o, _ = self.env.reset()
                    episode_steps = 0
                    current_ep_start = len(obs_buf)
                    episode_count += 1
                    continue
                
                # Calculate imitation reward
                try:
                    d_score = self.disc(obs, action)
                    r_imit = -torch.log(1 - d_score + 1e-8)
                    
                    # Cap imitation reward for stability
                    r_imit = torch.clamp(r_imit, -10.0, 10.0)
                except Exception as e:
                    logger.warning(f"Exception in discriminator: {e}")
                    r_imit = torch.tensor(0.0, device=self.device)
                
                # Combined reward
                r_total = r_imit + self.env_coef * r_env
                
                # Store in buffer
                obs_buf.append(obs)
                act_buf.append(action)
                ret_buf.append(r_total)
                val_buf.append(v)
                logp_buf.append(logp)
                done_buf.append(done)
                ep_start_buf.append(current_ep_start)  # Store episode start index
                
                # Update tracking variables
                episode_steps += 1
                total_steps += 1
                
                # Handle episode termination or max steps
                if done or episode_steps >= max_steps_per_episode:
                    o, _ = self.env.reset()
                    episode_steps = 0
                    current_ep_start = len(obs_buf)
                    episode_count += 1
                else:
                    o = o2
            
            # Set model back to training mode
            self.ac.train()
            
            # Print warning if NaNs were detected
            if nan_found:
                logger.warning("NaN values detected during trajectory collection")
        
        # Log collection statistics
        logger.info(f"Collected {total_steps} steps across {episode_count} episodes")
        
        # Stack tensors
        obs_buf = torch.stack(obs_buf)
        act_buf = torch.stack(act_buf)
        ret_buf = torch.stack(ret_buf)
        val_buf = torch.stack(val_buf)
        logp_buf = torch.stack(logp_buf)
        done_buf = torch.tensor(done_buf, dtype=torch.float32, device=self.device)
        ep_start_buf = torch.tensor(ep_start_buf, dtype=torch.long, device=self.device)
        
        # Log buffer shapes
        logger.debug(f"Trajectory collection: obs shape={obs_buf.shape}, act shape={act_buf.shape}")
        
        # Check for NaNs in all buffers and fix them
        for name, buf in [('obs', obs_buf), ('act', act_buf), ('ret', ret_buf), 
                         ('val', val_buf), ('logp', logp_buf)]:
            if torch.isnan(buf).any():
                nan_count = torch.isnan(buf).sum().item()
                logger.warning(f"NaN values in {name} buffer: {nan_count}/{buf.numel()} values")
                buf[torch.isnan(buf)] = 0.0
        
        # Compute GAE advantages and returns with proper episode boundaries
        adv_buf = torch.zeros_like(ret_buf)
        
        # Process each episode separately
        for i in range(len(obs_buf)):
            # Check if this is the last step in an episode
            is_last_in_ep = (i == len(obs_buf) - 1) or done_buf[i] or (ep_start_buf[i+1] != ep_start_buf[i] if i < len(obs_buf) - 1 else True)
            
            if is_last_in_ep:
                # For terminal states, only use the reward with no bootstrap
                adv_buf[i] = ret_buf[i] - val_buf[i]
            else:
                # Use bootstrapped return estimate
                adv_buf[i] = ret_buf[i] + self.gamma * val_buf[i+1] - val_buf[i]
        
        # Compute GAE for each episode with proper boundaries
        for ep_start in torch.unique(ep_start_buf, sorted=True):
            # Find end of this episode
            ep_end = len(obs_buf) - 1
            for j in range(ep_start, len(obs_buf)):
                if j == len(obs_buf) - 1 or done_buf[j] or (j+1 < len(ep_start_buf) and ep_start_buf[j+1] != ep_start_buf[j]):
                    ep_end = j
                    break
            
            # Compute GAE for this episode
            last_gae = 0
            for t in reversed(range(ep_start, ep_end + 1)):
                # For the last step, there's no bootstrap
                if t == ep_end:
                    # Terminal state has no future reward beyond immediate reward
                    delta = ret_buf[t] - val_buf[t]
                else:
                    # Bootstrap with next state value
                    delta = ret_buf[t] + self.gamma * val_buf[t+1] - val_buf[t]
                
                # Clip delta for stability
                delta = torch.clamp(delta, -10.0, 10.0)
                
                # Update GAE
                last_gae = delta + self.gamma * self.lam * last_gae
                adv_buf[t] = last_gae
        
        # Compute returns from advantages
        ret_buf = adv_buf + val_buf
        
        # Final check for NaNs in advantages and returns
        if torch.isnan(adv_buf).any():
            nan_count = torch.isnan(adv_buf).sum().item()
            logger.warning(f"NaN values in advantages: {nan_count}/{adv_buf.numel()} values")
            adv_buf[torch.isnan(adv_buf)] = 0.0
            
        if torch.isnan(ret_buf).any():
            nan_count = torch.isnan(ret_buf).sum().item()
            logger.warning(f"NaN values in returns: {nan_count}/{ret_buf.numel()} values")
            ret_buf[torch.isnan(ret_buf)] = 0.0
        
        return dict(obs=obs_buf, act=act_buf, logp=logp_buf, adv=adv_buf, ret=ret_buf, ep_starts=ep_start_buf)

    def update_discriminator(self, policy_data, batch_size):
        """Update GAIL discriminator to distinguish between expert and policy behaviors."""
        # Ensure policy data is float32 and on CPU for DataLoader
        obs_p, act_p = policy_data['obs'].cpu().to(dtype=torch.float32), policy_data['act'].cpu().to(dtype=torch.float32)
        
        # Create policy dataset with labels = 0 (not expert)
        policy_ds = TensorDataset(
            obs_p.to(self.device), 
            act_p.to(self.device), 
            torch.zeros(len(obs_p), device=self.device)
        )
        
        # Combine with pre-created expert loader for faster processing
        policy_loader = DataLoader(policy_ds, batch_size, shuffle=True, drop_last=False)
        
        # Train discriminator
        self.disc.train()
        total_loss = 0
        batches = 0
        
        # Train on policy data
        for p_batch in policy_loader:
            # Get a batch of expert data
            try:
                e_batch = next(self.expert_iter)
            except (StopIteration, AttributeError):
                # Restart the iterator if it's exhausted or doesn't exist
                self.expert_iter = iter(self.expert_loader)
                e_batch = next(self.expert_iter)
            
            # Combine expert and policy data
            o_b = torch.cat([e_batch[0], p_batch[0]], dim=0)
            a_b = torch.cat([e_batch[1], p_batch[1]], dim=0)
            y_b = torch.cat([e_batch[2], p_batch[2]], dim=0)
            
            # Train discriminator
            pred = self.disc(o_b, a_b)
            loss = nn.BCELoss()(pred, y_b)
            total_loss += loss.item()
            batches += 1
            
            self.disc_opt.zero_grad() 
            loss.backward()  # No retain_graph needed
            self.disc_opt.step()
        
        # Return average loss for logging
        return total_loss / max(1, batches)

    def update_ppo(self, data, ppo_epochs=10, mini_batch=256):
        obs, act, adv, ret, logp_old = data['obs'], data['act'], data['adv'], data['ret'], data['logp']
        
        # Replace NaN values in the data
        for tensor_name, tensor in [('obs', obs), ('act', act), ('adv', adv), ('ret', ret), ('logp_old', logp_old)]:
            nan_mask = torch.isnan(tensor)
            if nan_mask.any():
                nan_count = torch.isnan(tensor).sum().item()
                logger.warning(f"NaN values in {tensor_name}: {nan_count}/{tensor.numel()} values")
                tensor.data[nan_mask] = 0.0
        
        # Normalize advantages (after replacing NaNs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # Track metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_batches = 0
        
        self.ac.train()
        for epoch in range(ppo_epochs):
            idxs = torch.randperm(len(obs))
            for start in range(0, len(obs), mini_batch):
                total_batches += 1
                b = idxs[start:start+mini_batch]
                o_b, a_b, adv_b, ret_b, lp_b = obs[b], act[b], adv[b], ret[b], logp_old[b]
                
                # Forward pass with learned std dev
                pi_mu, std_dev, v_b = self.ac(o_b)
                
                # Check for NaN values in the network outputs
                if torch.isnan(pi_mu).any() or torch.isnan(v_b).any() or torch.isnan(std_dev).any():
                    logger.warning(f"NaNs in network output at epoch {epoch}, batch {start//mini_batch}")
                    logger.warning(f"pi_mu NaNs: {torch.isnan(pi_mu).sum().item()}/{pi_mu.numel()}")
                    logger.warning(f"std_dev NaNs: {torch.isnan(std_dev).sum().item()}/{std_dev.numel()}")
                    logger.warning(f"v_b NaNs: {torch.isnan(v_b).sum().item()}/{v_b.numel()}")
                    
                    # Replace NaN values with zeros
                    pi_mu = torch.nan_to_num(pi_mu, nan=0.0)
                    std_dev = torch.nan_to_num(std_dev, nan=0.1)  # Default to 0.1 for std_dev
                    v_b = torch.nan_to_num(v_b, nan=0.0)
                
                # Create Normal distribution with learned std_dev
                dist = torch.distributions.Normal(pi_mu, std_dev)
                
                # Compute log probabilities
                logp_b = dist.log_prob(a_b).sum(axis=-1)
                
                # PPO objective
                ratio = torch.exp(torch.clamp(logp_b - lp_b, min=-20, max=20))  # Clip for numerical stability
                
                # Clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv_b
                loss_pi = -torch.min(surr1, surr2).mean()
                
                # Value loss
                loss_v = 0.5 * (ret_b - v_b).pow(2).mean()
                
                # Total loss
                total_loss = loss_pi + loss_v
                
                # Check if loss is valid
                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                    logger.warning(f"Invalid loss value: {total_loss.item()}")
                    continue  # Skip this batch
                
                # Perform gradient step
                self.ac_opt.zero_grad()
                total_loss.backward()  # No retain_graph needed
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=1.0)
                
                # Check for invalid gradients
                for name, param in self.ac.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            nan_count = torch.isnan(param.grad).sum().item()
                            inf_count = torch.isinf(param.grad).sum().item()
                            logger.warning(f"Invalid gradients in {name}: {nan_count} NaNs, {inf_count} Infs")
                            param.grad = torch.zeros_like(param.grad)
                
                self.ac_opt.step()
                
                # Track loss values
                total_policy_loss += loss_pi.item()
                total_value_loss += loss_v.item()
        
        # Return average policy loss for logging
        return total_policy_loss / max(1, total_batches)

    def train(self, total_steps, batch_size, max_ep_steps=1000):
        """Train the GAIL-PPO agent for the specified number of steps."""
        steps = 0
        episodes = 0
        iterations = 0
        best_avg_episode_length = 0
        
        # Initialize expert data iterator
        self.expert_iter = iter(self.expert_loader)
        
        # Progress tracking
        start_time = time.time()
        progress_bar = tqdm(total=total_steps, desc="Training Progress")
        
        # Stats tracking
        episode_lengths = []
        discriminator_losses = []
        policy_losses = []
        
        # Print save directory information
        save_dir_abs = os.path.abspath(self.args.save_dir)
        logger.info(f"Models will be saved to: {save_dir_abs}")
        logger.info(f"Starting training for {total_steps} total steps")
        logger.info(f"Batch size: {batch_size}, Max episode steps: {max_ep_steps}")
        
        while steps < total_steps:
            iterations += 1
            
            # Collect trajectories
            logger.debug(f"Collecting trajectories (iteration {iterations})")
            data = self.collect_trajectories(batch_size, max_ep_steps)
            
            # Update tracking variables
            steps_this_iter = len(data['obs'])
            steps += steps_this_iter
            episodes_this_iter = len(torch.unique(data['ep_starts']))
            episodes += episodes_this_iter
            avg_ep_length = steps_this_iter / max(1, episodes_this_iter)
            episode_lengths.append(avg_ep_length)
            
            # Update discriminator
            logger.debug(f"Updating discriminator (iteration {iterations})")
            disc_loss = self.update_discriminator(data, batch_size)
            discriminator_losses.append(disc_loss)
            
            # Update PPO
            logger.debug(f"Updating policy with PPO (iteration {iterations})")
            policy_loss = self.update_ppo(data, self.args.ppo_epochs, self.args.mini_batch_size)
            policy_losses.append(policy_loss)
            
            # Update progress bar
            progress_bar.update(steps_this_iter)
            
            # Log progress
            elapsed = time.time() - start_time
            steps_per_sec = steps / elapsed
            remaining = (total_steps - steps) / max(1e-8, steps_per_sec)
            
            # Save best model if episode length is improving
            if avg_ep_length > best_avg_episode_length:
                best_avg_episode_length = avg_ep_length
                if self.args.save_model:
                    self.save_model("best_model.pt")
            
            # Log training progress
            if iterations % 10 == 0 or steps >= total_steps:
                logger.info(
                    f"Iteration {iterations} | "
                    f"Steps: {steps}/{total_steps} ({steps/total_steps*100:.1f}%) | "
                    f"Episodes: {episodes} | "
                    f"Avg Episode Length: {avg_ep_length:.1f} steps | "
                    f"Disc Loss: {disc_loss:.4f} | "
                    f"ETA: {int(remaining/60)}m {int(remaining%60)}s"
                )
                
                # Save checkpoint if requested
                if self.args.save_model and iterations % 50 == 0:
                    self.save_model(f"checkpoint_{iterations}.pt")
        
        # Save final model
        if self.args.save_model:
            self.save_model("final_model.pt")
            
        progress_bar.close()
        logger.info(f"Training completed in {elapsed/60:.1f} minutes")
        logger.info(f"Final stats: {episodes} episodes, avg length {sum(episode_lengths[-10:])/10:.1f} steps")
        
        return {
            "steps": steps,
            "episodes": episodes,
            "iterations": iterations,
            "episode_lengths": episode_lengths,
            "discriminator_losses": discriminator_losses,
            "policy_losses": policy_losses,
            "training_time": elapsed
        }
        
    def save_model(self, filename):
        """Save the model to a file."""
        save_path = os.path.join(self.args.save_dir, filename)
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        state = {
            'policy_state': self.ac.state_dict(),
            'discriminator_state': self.disc.state_dict(),
            'args': self.args,
        }
        torch.save(state, save_path)
        abs_path = os.path.abspath(save_path)
        logger.info(f"Model saved to {save_path}")
        logger.info(f"ABSOLUTE PATH: {abs_path}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer', required=True, help='Expert buffer .pth')
    parser.add_argument('--init_policy', required=False, default=None, help='DAgger-pretrained policy .pth')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[256,256])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ac_wd', type=float, default=1e-5)
    parser.add_argument('--disc_lr', type=float, default=1e-4)
    parser.add_argument('--disc_wd', type=float, default=1e-6)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--env_coef', type=float, default=0.1,
                        help='Weight for environment reward')
    parser.add_argument('--max_ep_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--learn_std', action='store_true', 
                        help='Whether to learn the policy standard deviation')
    parser.add_argument('--log_std_min', type=float, default=-20.0,
                        help='Minimum log standard deviation')
    parser.add_argument('--log_std_max', type=float, default=2.0,
                        help='Maximum log standard deviation')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help='Number of PPO epochs per batch')
    parser.add_argument('--mini_batch_size', type=int, default=256,
                        help='PPO mini-batch size')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model checkpoints')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='Directory to save models')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Logging level')
    parser.add_argument('--total_steps', type=int, default=1_000_000)
    parser.add_argument('--batch_size', type=int, default=2048)
    # Check CUDA availability before setting default device
    cuda_available = torch.cuda.is_available()
    default_device = 'cuda' if cuda_available else 'cpu'
    parser.add_argument('--device', choices=['cpu','cuda'], default=default_device)
    args = parser.parse_args()
    
    # Add safety check for CUDA device
    if args.device == 'cuda' and not cuda_available:
        print("Warning: CUDA device requested but torch.cuda.is_available() is False.")
        print("Falling back to CPU device.")
        args.device = 'cpu'
    
    # Set logging level based on arguments
    if args.log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif args.log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif args.log_level == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif args.log_level == 'ERROR':
        logger.setLevel(logging.ERROR)
    
    env = make_g1_env()
    trainer = GAILPPO(env, args.buffer, args.init_policy, args)
    trainer.train(args.total_steps, args.batch_size)
