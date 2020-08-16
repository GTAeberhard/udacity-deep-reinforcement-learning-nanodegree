import copy
import random
import numpy as np

import torch

from .hyperparameters import LEARNING_RATE, LEARNING_RATE_CRITIC, TAU, WEIGHT_DECAY
from .network import DDPGActorNetwork, MultiAgentDDPGCriticNetwork

class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents, train=False, seed=5, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.train = train

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.actor_network = DDPGActorNetwork(self.state_size, self.action_size).to(self.device)

        if self.train:
            self.critic_network = MultiAgentDDPGCriticNetwork(
                self.state_size, self.action_size, self.num_agents
            ).to(self.device)

            self.optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)
            self.optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE_CRITIC,
                                                     weight_decay=WEIGHT_DECAY)

            self.actor_network.train()
            self.critic_network.train()

            self.target_actor_network = DDPGActorNetwork(self.state_size, self.action_size).to(self.device)
            self.target_critic_network = MultiAgentDDPGCriticNetwork(
                self.state_size, self.action_size, self.num_agents
            ).to(self.device)

            self.hard_update(self.actor_network, self.target_actor_network)
            self.hard_update(self.critic_network, self.target_critic_network)

            self.noise = OUNoise(self.action_size, 0)
        else:
            self.actor_network.eval()
            self.critic_network = None
            self.noise = None

    def act(self, states, eps=1.0):
        self.actor_network.eval()
        with torch.no_grad():
            actions = self.actor_network(states)
            if self.train:
                noise = torch.from_numpy(self.noise.sample()).float().unsqueeze(0).to(self.device)
                # noise = torch.from_numpy(0.5 * np.random.rand).float().unsqueeze(0).to(self.device)
                actions += eps * noise
            actions.clamp_(-1, 1).squeeze(0).cpu().numpy()
        if self.train:
            self.actor_network.train()

        return actions

    def soft_update_networks(self):
        self.soft_update(self.actor_network, self.target_actor_network, TAU)
        self.soft_update(self.critic_network, self.target_critic_network, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    @staticmethod
    def hard_update(local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def reset(self):
        if self.noise:
            self.noise.reset()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
