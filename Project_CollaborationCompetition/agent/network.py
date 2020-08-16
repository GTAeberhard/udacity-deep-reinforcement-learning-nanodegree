import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .hyperparameters import DNN_ARCHITECTURE


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class DDPGActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0, hidden_layers=DNN_ARCHITECTURE):
        super(DDPGActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Input layer
        self.input_layer = nn.Linear(state_size, hidden_layers[0])

        # Hidden layers
        self.hidden_layers = []
        for current_layer, next_layer in zip(hidden_layers, hidden_layers[1:]):
            self.hidden_layers.append(nn.Linear(current_layer, next_layer))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        # Output action layer
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return F.tanh(self.output_layer(x))


class MultiAgentDDPGCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_agents, seed=0, hidden_layers=DNN_ARCHITECTURE):
        assert(len(hidden_layers) >= 2), "The DDPG critic network requires at least two hidden layers to be defined."
        super(MultiAgentDDPGCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Input layers
        self.state_input_layer = nn.Linear(state_size * num_agents, hidden_layers[0])
        self.action_input_layer = nn.Linear(hidden_layers[0] + action_size * num_agents, hidden_layers[1])
        self.bn_action_input = nn.BatchNorm1d(hidden_layers[1])

        # Hidden layers
        self.hidden_layers = []
        if len(hidden_layers) > 2:
            for current_layer, next_layer in zip(hidden_layers[1:], hidden_layers[2:]):
                self.hidden_layers.append(nn.Linear(current_layer, next_layer))
            self.hidden_layers = nn.ModuleList(self.hidden_layers)

        # Output action-value layer
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

    def reset_parameters(self):
        self.state_input_layer.weight.data.uniform_(*hidden_init(self.state_input_layer))
        self.action_input_layer.weight.data.uniform_(*hidden_init(self.action_input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.relu(self.state_input_layer(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.bn_action_input(self.action_input_layer(x)))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)
