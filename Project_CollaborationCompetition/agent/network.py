"""
Neural networks for the actor and critic as part of the DDPG algorithm, implemented
in PyTorch using nn.Module.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .hyperparameters import DEFAULT_HYPERPARAMETERS


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class DDPGActorNetwork(nn.Module):
    """
    Neural network for the actor in the DDPG algorithm. The actor is used to generate continuous
    actions given the current state of the environment.

    Attributes
    ----------
    input_layer: Input layer of the network, implemented as linear nodes
    hidden_layers: nn.ModuleList which contains the hidden layers of the neural networks, implemented as linear nodes
    output_layer: Output layer of the network, implemented as linear nodes
    seed: random seed to be used within PyTorch 
    """
    def __init__(self, state_size, action_size, seed=0, hidden_layers=DEFAULT_HYPERPARAMETERS["DNN_ARCHITECTURE"]):
        """
        Initialized the layers of the neural network.

        Parameters
        ----------
        state_size: size of the input state space for the network
        action_size: size of the output action space for the network
        hidden_layers: a list which defines the number and size of the hidden layers of the neural network
        seed: random seed to be used within PyTorch
        """
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
        """Reset the weights of the network to an initial state."""
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Implements the forward pass of the neural network using ReLu activation functions for the input and hidden
        layers and a tanh activation function for the output layer.

        Parameters
        ----------
        state: input state to the neural network

        Returns
        -------
        actions: output actions from the neural network
        """
        x = F.relu(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return F.tanh(self.output_layer(x))


class MultiAgentDDPGCriticNetwork(nn.Module):
    """
    Neural network for the critic in the multi-agent DDPG algorithm. The critic is used to evaluate how good a
    state-action pair is, i.e. estimates the action-value function. Note that in the multi-agent case, the state and
    actions of all agents is given to the critic network.

    Attributes
    ----------
    state_input_layer: Input layer of the neural network for the states of the agent, implemented as linear nodes.
    action_input_layer: Input layer of the neural network for the actions of the agent. This layer is actually the 2nd
        layer in the network and get concatenated with the first state layer. Implemented as linear nodes.
    bn_action_input: Batch normalization layer after the action input layer
    hidden_layers: nn.ModuleList which contains the hidden layers of the network, implemented as linear nodes.
    output_layer: Output layer of the network, implemented as linear nodes
    seed: random seed to be used within PyTorch
    """
    def __init__(self, state_size, action_size, num_agents, seed=0,
                 hidden_layers=DEFAULT_HYPERPARAMETERS["DNN_ARCHITECTURE"]):
        """
        Initialized the layers of the neural network.

        Parameters
        ----------
        state_size: size of the input state space for the network
        action_size: size of the output action space for the network
        num_agents: number of agents in the environment
        hidden_layers: a list which defines the number and size of the hidden layers of the neural network. Note that
            at least two hidden layers must be defined, since the action input layer is also considered to be the first
            hidden layer.
        seed: random seed to be used within PyTorch
        """
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
        """Reset the weights of the network to an initial state."""
        self.state_input_layer.weight.data.uniform_(*hidden_init(self.state_input_layer))
        self.action_input_layer.weight.data.uniform_(*hidden_init(self.action_input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Implements the forward pass of the neural network using ReLu activation functions for the input and hidden
        layers and no activation function for the output layer.

        Parameters
        ----------
        state: input states to the neural network (for all agents in the environment)
        state: input actions to the neural network (for all agents in the environment)

        Returns
        -------
        action-value: Value of the current state-action pairs for all of the agents in the environment
        """
        xs = F.relu(self.state_input_layer(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.bn_action_input(self.action_input_layer(x)))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        return self.output_layer(x)
