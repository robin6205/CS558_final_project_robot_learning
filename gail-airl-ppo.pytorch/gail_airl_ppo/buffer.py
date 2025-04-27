import os
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        if self._n < self.buffer_size:
            print(f"Warning: Buffer not fully filled. Using available {self._n} samples instead of {self.buffer_size}.")
            return (
                self.states[:self._n],
                self.actions[:self._n],
                self.rewards[:self._n],
                self.dones[:self._n],
                self.log_pis[:self._n],
                self.next_states[:self._n]
            )
        
        if self._p >= self.buffer_size:
            start = self._p - self.buffer_size
            end = self._p
        else:
            start_end = self.total_size - (self.buffer_size - self._p)
            end_end = self.total_size
            start_begin = 0
            end_begin = self._p
            
            return (
                torch.cat([self.states[start_end:end_end], self.states[start_begin:end_begin]]),
                torch.cat([self.actions[start_end:end_end], self.actions[start_begin:end_begin]]),
                torch.cat([self.rewards[start_end:end_end], self.rewards[start_begin:end_begin]]),
                torch.cat([self.dones[start_end:end_end], self.dones[start_begin:end_begin]]),
                torch.cat([self.log_pis[start_end:end_end], self.log_pis[start_begin:end_begin]]),
                torch.cat([self.next_states[start_end:end_end], self.next_states[start_begin:end_begin]])
            )
        
        return (
            self.states[start:end],
            self.actions[start:end],
            self.rewards[start:end],
            self.dones[start:end],
            self.log_pis[start:end],
            self.next_states[start:end]
        )

    def sample(self, batch_size):
        sample_upper_bound = min(self._n, self.total_size)
        
        if sample_upper_bound < batch_size:
            print(f"Warning: Not enough samples in buffer ({sample_upper_bound}). Sampling with replacement.")
            idxes = np.random.choice(sample_upper_bound, size=batch_size, replace=True)
        else:
            idxes = np.random.randint(low=0, high=sample_upper_bound, size=batch_size)
            
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
