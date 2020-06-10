import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.hyperparameters import DNN_ARCHITECTURE


class PolicyNetwork(nn.Module):
    """DNN for the policy"""

    def __init__(self, state_size, action_size, seed=0, hidden_layers=DNN_ARCHITECTURE):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list[int]): Size of fully connected hidden layers
        """
        super(PolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        assert(len(hidden_layers) > 0), "Must specify at least a single hidden layer for the neutral network."

        # Input layer
        self.input_layer = nn.Linear(state_size, hidden_layers[0])

        # Hidden layers
        self.hidden_layers = []
        for current_layer, next_layer in zip(hidden_layers, hidden_layers[1:]):
            self.hidden_layers.append(nn.Linear(current_layer, next_layer))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        # Output layer
        self.output_layer_mean = nn.Linear(hidden_layers[-1], action_size)
        self.output_layer_var = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return F.tanh(self.output_layer_mean(x)), F.softplus(self.output_layer_var(x))
