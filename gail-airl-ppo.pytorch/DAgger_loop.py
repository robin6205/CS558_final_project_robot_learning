# dagger_loop.py
# DAgger implementation using nearest-neighbor expert oracle on expert buffer

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from g1_env import make_g1_env
from bc_pretrain import build_mlp

class DAgger:
    def __init__(self, 
                 env_name,
                 buffer_path,
                 policy_path,
                 hidden_sizes,
                 lr,
                 batch_size,
                 weight_decay,
                 device):
        # Load expert buffer for oracle
        data = torch.load(buffer_path)
        self.expert_states  = data['state'].numpy()    # (N, dim)
        self.expert_actions = data['action'].numpy()   # (N, dim)

        # Build policy network
        input_dim  = self.expert_states.shape[1]
        output_dim = self.expert_actions.shape[1]
        self.policy = build_mlp(input_dim, output_dim, hidden_sizes).to(device)
        self.policy.load_state_dict(torch.load(policy_path))
        self.device = device

        # Optimizer for aggregation retraining
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

        # Dataloader settings
        self.batch_size = batch_size
        # Initialize aggregated dataset with expert buffer
        dataset = TensorDataset(torch.tensor(self.expert_states, dtype=torch.float32),
                                torch.tensor(self.expert_actions, dtype=torch.float32))
        self.agg_dataset = dataset

        # Environment
        self.env = make_g1_env()
        
        # Verify environment and policy dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        policy_in_dim = self.expert_states.shape[1]
        policy_out_dim = self.expert_actions.shape[1]
        
        print(f"Environment state dim: {state_dim}, policy input dim: {policy_in_dim}")
        print(f"Environment action dim: {action_dim}, policy output dim: {policy_out_dim}")
        
        assert state_dim == policy_in_dim, f"Environment state dim ({state_dim}) does not match policy input dim ({policy_in_dim})"
        assert action_dim == policy_out_dim, f"Environment action dim ({action_dim}) does not match policy output dim ({policy_out_dim})"

    def nearest_expert_action(self, state):
        # Compute L2 distances to all expert states (qpos+qvel)
        diffs = self.expert_states - state
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.argmin(dists)
        return self.expert_actions[idx]

    def collect_data(self, num_rollouts, horizon):
        new_states, new_actions = [], []
        for _ in range(num_rollouts):
            reset_result = self.env.reset()
            # Handle tuple return from reset (observation, info)
            if isinstance(reset_result, tuple):
                s = reset_result[0]
            else:
                s = reset_result
            for t in range(horizon):
                state_np = s.astype(np.float32)
                
                with torch.no_grad():
                    a = self.policy(torch.tensor(state_np).to(self.device)).cpu().numpy()
                
                # Query oracle expert action
                a_star = self.nearest_expert_action(state_np)
                new_states.append(state_np)
                new_actions.append(a_star)
                step_result = self.env.step(a)
                # Handle tuple return from step with varying lengths
                if isinstance(step_result, tuple):
                    s = step_result[0]
                    done = step_result[2] if len(step_result) <= 4 else (step_result[2] or step_result[3])
                else:
                    s, _, done, _ = step_result
                if done:
                    break
        return np.array(new_states), np.array(new_actions)

    def aggregate_and_train(self, states, actions, epochs):
        # Create dataset from collected
        collected_ds = TensorDataset(torch.tensor(states, dtype=torch.float32),
                                     torch.tensor(actions, dtype=torch.float32))
        self.agg_dataset = ConcatDataset([self.agg_dataset, collected_ds])
        loader = DataLoader(self.agg_dataset, batch_size=self.batch_size, shuffle=True)
        # Train policy on aggregated
        self.policy.train()
        for ep in range(1, epochs+1):
            total_loss = 0
            for s_batch, a_batch in loader:
                s_batch, a_batch = s_batch.to(self.device), a_batch.to(self.device)
                pred = self.policy(s_batch)
                loss = self.criterion(pred, a_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * s_batch.size(0)
            avg = total_loss / len(loader.dataset)
            print(f"DAgger Retrain Epoch {ep}/{epochs}  Loss: {avg:.6f}")
        # Save updated policy
        torch.save(self.policy.state_dict(), 'dagger_policy.pth')

    def run(self, iterations, rollouts, horizon, retrain_epochs):
        for it in range(1, iterations+1):
            print(f"\n=== DAgger Iteration {it}/{iterations} ===")
            # 1. Collect data from current policy
            states, actions = self.collect_data(rollouts, horizon)
            print(f"Collected {len(states)} state-action pairs.")
            # 2. Aggregate and retrain
            self.aggregate_and_train(states, actions, retrain_epochs)

if __name__=='__main__':
    parser = argparse.ArgumentParser("DAgger Loop for G1 Sidestep")
    parser.add_argument('--env', type=str, default='G1Env-v0')
    parser.add_argument('--buffer', required=True, help='Expert buffer path')
    parser.add_argument('--policy', required=True, help='Pretrained BC policy path')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[256,256])
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--iters', type=int, default=5, help='DAgger iterations')
    parser.add_argument('--rollouts', type=int, default=10, help='Rollouts per iteration')
    parser.add_argument('--horizon', type=int, default=200, help='Max steps per rollout')
    parser.add_argument('--epochs', type=int, default=20, help='Retrain epochs per iteration')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    trainer = DAgger(
        env_name=args.env,
        buffer_path=args.buffer,
        policy_path=args.policy,
        hidden_sizes=args.hidden_sizes,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        device=torch.device(args.device)
    )
    trainer.run(
        iterations=args.iters,
        rollouts=args.rollouts,
        horizon=args.horizon,
        retrain_epochs=args.epochs
    )
