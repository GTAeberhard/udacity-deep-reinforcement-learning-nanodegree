import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .hyperparameters import BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE, GAMMA, TAU, UPDATE_EVERY, ALPHA

from .q_network import QNetwork
from .replay_buffers import ReplayBuffer, PriorityReplayBuffer


class DqnAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double_dqn=False, priority_replay=False, dueling_dqn=False,
                 device=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)

        # Replay memory
        if priority_replay:
            self.memory = PriorityReplayBuffer(action_size, BUFFER_SIZE, seed, alpha=ALPHA, device=self.device)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, seed, device=self.device)

        # Initialize episode step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Use DQN extensions
        self.double_dqn = double_dqn

    def step(self, state, action, reward, next_state, done, beta=1.0):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, batch_indices, experience_weights = self.memory.sample(BATCH_SIZE, beta)
                td_error = self._learn(experiences, experience_weights, GAMMA)
                if isinstance(self.memory, PriorityReplayBuffer):
                    td_error = td_error + 1e-5
                    self.memory.update_priorities(batch_indices, td_error)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def _learn(self, experiences, experience_weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.double_dqn:
            actions_qnetwork_local = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, actions_qnetwork_local)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        td_error = Q_expected - Q_targets
        losses = experience_weights * (td_error ** 2)
        loss = losses.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self._soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        return np.abs(td_error.data.cpu().squeeze(1).numpy())

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
