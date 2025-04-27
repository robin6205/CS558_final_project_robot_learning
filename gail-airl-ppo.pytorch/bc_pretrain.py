# bc_pretrain.py
# Supervised pretraining (Behavior Cloning) on expert buffer for G1 sidestep

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

def build_mlp(input_dim, output_dim, hidden_sizes, activation=nn.ReLU):
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)

class BCTrainer:
    def __init__(self, buffer_path, hidden_sizes, lr, batch_size, weight_decay, device):
        data = torch.load(buffer_path)
        self.states  = data['state'].to(dtype=torch.float32)
        self.actions = data['action'].to(dtype=torch.float32)
        self.device  = device

        dataset = TensorDataset(self.states, self.actions)
        # 80/20 train/val split
        n = len(dataset)
        n_val = int(0.2 * n)
        n_train = n - n_val
        self.train_ds, self.val_ds = random_split(dataset, [n_train, n_val])

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_ds,   batch_size=batch_size)

        input_dim  = self.states.shape[1]
        output_dim = self.actions.shape[1]
        self.policy = build_mlp(input_dim, output_dim, hidden_sizes).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, epochs):
        best_val = float('inf')
        for ep in range(1, epochs+1):
            # training loop
            self.policy.train()
            total_loss = 0
            for s,a in self.train_loader:
                s,a = s.to(self.device), a.to(self.device)
                pred = self.policy(s)
                loss = self.criterion(pred, a)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * s.size(0)
            avg_train = total_loss / len(self.train_loader.dataset)

            # validation
            self.policy.eval()
            val_loss = 0
            with torch.no_grad():
                for s,a in self.val_loader:
                    s,a = s.to(self.device), a.to(self.device)
                    val_loss += self.criterion(self.policy(s), a).item() * s.size(0)
            avg_val = val_loss / len(self.val_loader.dataset)

            print(f"Epoch {ep}/{epochs}  Train Loss: {avg_train:.6f}  Val Loss: {avg_val:.6f}")
            # save best
            if avg_val < best_val:
                best_val = avg_val
                torch.save(self.policy.state_dict(), 'bc_policy_best.pth')
        print(f"Best validation loss: {best_val:.6f}. Saved bc_policy_best.pth")

if __name__=='__main__':
    parser = argparse.ArgumentParser("Behavior Cloning Pretraining for G1 Sidestep")
    parser.add_argument('--buffer', required=True, help='Path to expert buffer .pth')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[256,256], help='MLP hidden layer sizes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    trainer = BCTrainer(
        buffer_path   = args.buffer,
        hidden_sizes  = args.hidden_sizes,
        lr             = args.lr,
        batch_size     = args.batch_size,
        weight_decay   = args.weight_decay,
        device         = torch.device(args.device)
    )
    trainer.train(args.epochs)
