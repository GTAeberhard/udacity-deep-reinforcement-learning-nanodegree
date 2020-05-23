import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list[int]): Size of fully connected hidden layers
        """
        super(QNetwork, self).__init__()
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
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)
