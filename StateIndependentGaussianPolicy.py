import torch
from torch import nn

from gail_airl_ppo.network.utils import build_mlp, reparameterize, evaluate_lop_pi


class StateIndependentGaussianPolicy(nn.Module):
    """
    A state-independent Gaussian policy that takes a state as input and outputs
    an action sampled from a Gaussian distribution.
    
    The policy consists of:
    1. A neural network that maps states to mean action values
    2. A learnable log standard deviation parameter
    
    The policy uses the reparameterization trick for sampling actions, which allows
    backpropagation through the sampling process.
    
    This version can optionally consider a history of states by maintaining a buffer
    of recent states and using them to make decisions.
    """

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(), history_length=1):
        super().__init__()
        
        self.history_length = history_length
        self.state_dim = state_shape[0]
        
        # If using history, the input dimension is multiplied by history_length
        input_dim = state_shape[0] * history_length
        
        # State history buffer (initialized as zeros)
        self.state_history = torch.zeros(history_length, state_shape[0])

        # Network to predict mean actions from states
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        
        # Learnable log standard deviation parameter
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def update_history(self, state):
        """
        Update state history buffer with a new state
        
        Args:
            state: New state to add to history
        """
        if self.history_length > 1:
            # Convert state to tensor if it's not already
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float)
            
            # Ensure state is the right shape
            if len(state.shape) > 1:
                state = state.squeeze(0)
                
            # Shift history back and add new state
            self.state_history = torch.cat([self.state_history[1:], state.unsqueeze(0)], dim=0)
    
    def get_features(self, state):
        """
        Get the features for the policy by combining current state with history
        
        Args:
            state: Current state
            
        Returns:
            Combined state features including history
        """
        if self.history_length == 1:
            return state
            
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float)
            
        # Handle batched and non-batched states
        batched = len(state.shape) > 1
        
        if batched:
            batch_size = state.shape[0]
            device = state.device
            
            # For each state in the batch, we need to create a history representation
            # This is a bit complex because each item in the batch might need its own history
            # For simplicity, we'll use the same history for all items in the batch
            # In a real implementation, you'd maintain separate histories for each environment
            
            # Repeat history for each item in the batch
            history = self.state_history.to(device).repeat(batch_size, 1, 1)
            
            # Replace the most recent history entry with the current state
            if self.history_length > 0:
                history[:, -1, :] = state
                
            # Flatten history for each batch item
            return history.view(batch_size, -1)
        else:
            # Non-batched version
            device = state.device
            
            # Create full history including current state
            full_history = torch.cat([
                self.state_history[-(self.history_length-1):].to(device),
                state.unsqueeze(0)
            ], dim=0)
            
            # Flatten history
            return full_history.view(-1)

    def forward(self, states):
        """
        Returns the deterministic action (mean of the distribution)
        
        Args:
            states: Batch of states
            
        Returns:
            Deterministic actions (tanh of means)
        """
        features = self.get_features(states)
        return torch.tanh(self.net(features))

    def sample(self, states):
        """
        Sample actions from the Gaussian distribution
        
        Args:
            states: Batch of states
            
        Returns:
            Tuple of (actions, log_probs)
        """
        features = self.get_features(states)
        actions, log_probs = reparameterize(self.net(features), self.log_stds)
        
        # Update history with the current state (only if not in a batch)
        if len(states.shape) == 1 and self.history_length > 1:
            self.update_history(states)
            
        return actions, log_probs

    def evaluate_log_pi(self, states, actions):
        """
        Evaluate log probability of actions given states
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Log probabilities of the actions
        """
        features = self.get_features(states)
        return evaluate_lop_pi(self.net(features), self.log_stds, actions) 