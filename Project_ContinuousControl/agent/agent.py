import random
import torch
from torch.distributions import Normal

from .network import PolicyNetwork


class Agent():
    """An agent that interacts with and learns from the environment using policy-based DRL methods."""

    def __init__(self, state_size, action_size, train=False, max_t=1000, seed=0, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.train = train

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.policy_network = PolicyNetwork(self.state_size, self.action_size).to(self.device)
        if self.train:
            self.policy_network.train()
            self.trajectory_log_probs = []
            self.trajectory_rewards = []

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        mean, variance = self.policy_network(state)
        actions_distribution = Normal(mean, variance)
        selected_actions = actions_distribution.sample()

        return selected_actions.cpu().data.numpy()

    # def trajectory_append(self, state, actions, reward):
    #     self.trajectory.append((state, actions))
    #     self.rewards.append = reward