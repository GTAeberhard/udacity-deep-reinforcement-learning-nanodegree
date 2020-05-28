import random
import torch
import numpy as np

from collections import namedtuple, deque


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed, device=None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size, beta=None):
        """Randomly sample a batch of experiences from memory."""
        experiences, indices = self._sample_experiences_from_probabilities(batch_size)
        weights = torch.ones(len(indices)).unsqueeze(1).to(self.device)
        return experiences, indices, weights

    def _sample_experiences_from_probabilities(self, batch_size, sample_probabilities=None):
        indices = np.random.choice(len(self.memory), batch_size, p=sample_probabilities)

        experiences = [self.memory[idx] for idx in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones), indices

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PriorityReplayBuffer(ReplayBuffer):
    """Fixed-size priority buffer to store experience tuples and sample them based on a priority value."""

    def __init__(self, action_size, buffer_size, seed, alpha=0.6, device=None):
        super().__init__(action_size, buffer_size, seed, device)

        self.alpha = alpha
        self.priorities = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.priorities.append(max(self.priorities) if len(self.memory) > 0 else 1.0)
        super().add(state, action, reward, next_state, done)

    def sample(self, batch_size, beta=1.0):
        memory_probabilities = np.array(list(self.priorities)) ** self.alpha
        memory_probabilities /= sum(memory_probabilities)

        experiences, indicies = self._sample_experiences_from_probabilities(batch_size, memory_probabilities)

        weights = (len(self.memory) * memory_probabilities[indicies]) ** (-beta)
        weights /= max(weights)

        return experiences, indicies, torch.tensor(weights).float().unsqueeze(1).to(self.device)

    def update_priorities(self, indicies, losses):
        for idx, new_priority in zip(indicies, losses):
            self.priorities[idx] = new_priority
