import copy
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn.functional as F

from .hyperparameters import (
    LEARNING_RATE, LEARNING_RATE_CRITIC, BATCH_SIZE, BUFFER_SIZE, GAMMA, GRADIENT_CLIPPING_MAX, TAU, WEIGHT_DECAY,
    STEPS_BETWEEN_LEARNING, LEARNING_ITERATIONS, EPSILON_START, EPSILON_END, EPSILON_DECAY
)
from .network import DDPGActorNetwork, DDPGCriticNetwork


class DDPGAgent():
    def __init__(self, state_size, action_size, train=False, seed=5, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.train = train
        self.steps = 0
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.eps = EPSILON_START

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.actor_network = DDPGActorNetwork(self.state_size, self.action_size).to(self.device)

        if self.train:
            self.critic_network = DDPGCriticNetwork(self.state_size, self.action_size).to(self.device)

            self.optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)
            self.optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE_CRITIC,
                                                     weight_decay=WEIGHT_DECAY)

            self.actor_network.train()
            self.critic_network.train()

            self.target_actor_network = DDPGActorNetwork(self.state_size, self.action_size).to(self.device)
            self.target_critic_network = DDPGCriticNetwork(self.state_size, self.action_size).to(self.device)

            # Make sure target and local networks have the same parameters at initialization
            self.hard_update(self.critic_network, self.target_critic_network)
            self.hard_update(self.actor_network, self.target_actor_network)

            self.noise = None
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device)
        else:
            self.critic_network = None
            self.actor_network.eval()

    def act(self, states_input):
        if self.train and self.noise is None:
            self.noise = OUNoise(self.action_size * len(states_input), 0)

        states = torch.from_numpy(states_input).float().unsqueeze(0).to(self.device)
        self.actor_network.eval()
        with torch.no_grad():
            actions = self.actor_network(states)
            if self.train:
                noise = torch.from_numpy(self.noise.sample()).float().reshape(
                    len(states_input), self.action_size
                ).unsqueeze(0).to(self.device)
                actions += self.eps * noise
            actions_output = actions.clamp(-1, 1).squeeze(0).cpu().numpy()
        if self.train:
            self.actor_network.train()

        return actions_output

    def step(self, states, actions, rewards, next_states, dones):
        self.steps += 1
        if self.train:
            self.memory.add(states, actions, rewards, next_states, dones)
            self.learn()
            self.eps = max(self.eps - EPSILON_DECAY, EPSILON_END)

    def learn(self, gamma=GAMMA):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if not (len(self.memory) > BATCH_SIZE and self.steps % STEPS_BETWEEN_LEARNING == 0):
            return

        for i in range(LEARNING_ITERATIONS):
            states, actions, rewards, next_states, dones = self.memory.sample()

            # TODO: Try to remove this?
            if rewards.sum() < 0.05:
                continue

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.target_actor_network(next_states)
            Q_targets_next = self.target_critic_network(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_network(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), GRADIENT_CLIPPING_MAX)
            self.optimizer_critic.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_network(states)
            actor_loss = -self.critic_network(states, actions_pred).mean()
            # Minimize the loss
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), GRADIENT_CLIPPING_MAX)
            self.optimizer_actor.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_network, self.target_critic_network, TAU)
            self.soft_update(self.actor_network, self.target_actor_network, TAU)

            self.last_actor_loss = actor_loss.item()
            self.last_critic_loss = critic_loss.item()

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
        if self.train and self.noise:
            self.noise.reset


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def add(self, states, actions, rewards, next_states, dones):
        """Add new experiences to memory."""
        assert(len(states) == len(actions) == len(rewards) == len(next_states == len(dones)))
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            e = self.experience(s, a, r, ns, d)
            self.memory.append(e)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.standard_normal() for i in range(len(x))])
        self.state = x + dx
        return self.state