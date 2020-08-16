import copy
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn.functional as F

from .hyperparameters import (
    LEARNING_RATE, LEARNING_RATE_CRITIC, BATCH_SIZE, BUFFER_SIZE, GAMMA, GRADIENT_CLIPPING_MAX, TAU, WEIGHT_DECAY,
    STEPS_BETWEEN_LEARNING, EPSILON_START, EPSILON_END, EPSILON_DECAY, WARM_UP_EPISODES
)
from .ddpg_agent import DDPGAgent
from .network import DDPGActorNetwork, MultiAgentDDPGCriticNetwork


class MultiAgentDDPG:
    def __init__(self, state_size, action_size, num_agents, train=False, seed=0, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.train = train
        self.steps = 0
        self.episodes = 0
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.eps = EPSILON_START

        np.random.seed(seed)
        torch.manual_seed(seed)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.agents = [
            DDPGAgent(self.state_size, self.action_size, self.num_agents, self.train, seed=seed, device=device)
            for _ in range(self.num_agents)
        ]

        if self.train:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed, self.device)

    def act(self, states_input):
        if self.episodes < WARM_UP_EPISODES:
            # Take a random action during the warm up phase to gather experience
            return np.random.uniform(
                low=-1, high=1, size=self.action_size * self.num_agents
            ).reshape(self.num_agents, self.action_size)
        else:
            actions_agent = torch.empty(self.num_agents, self.action_size).to(self.device)
            for i, a in enumerate(self.agents):
                states_agent = torch.from_numpy(states_input[i]).float().unsqueeze(0).to(self.device)
                actions_agent[i, :] = a.act(states_agent, self.eps)
            
            return actions_agent.squeeze(0).cpu().numpy()

    def step(self, states, actions, rewards, next_states, dones):
        self.steps += 1
        if self.train:
            self.memory.add(states, actions, rewards, next_states, dones)
            if self.episodes >= WARM_UP_EPISODES:
                self.learn()
                self.eps = max(self.eps - EPSILON_DECAY, EPSILON_END)
            if np.any(dones):
                self.episodes += 1

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

        self.last_actor_loss = np.zeros(2)
        self.last_critic_loss = np.zeros(2)

        states, actions, rewards, next_states, dones = self.memory.sample()
        
        for i, a in enumerate(self.agents):
            actions_next = actions.clone()
            actions_pred = actions.clone()
            for j, b in enumerate(self.agents):
                i_start_states = j * self.state_size
                i_end_states = i_start_states + self.state_size
                agent_states = states[:, i_start_states:i_end_states]
                agent_next_states = next_states[:, i_start_states:i_end_states]

                i_start_actions = j * self.action_size
                i_end_actions = i_start_actions + self.action_size
                actions_next[:, i_start_actions:i_end_actions] = b.target_actor_network(agent_next_states)
                actions_pred[:, i_start_actions:i_end_actions] = b.actor_network(agent_states)

            # Update the agent's critic
            # Get Q values from target models
            with torch.no_grad():
                Q_targets_next = a.target_critic_network(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            # Q_targets = rewards[:, i].unsqueeze(-1) + (gamma * Q_targets_next * (1 - dones[:, i].unsqueeze(-1)))
            Q_targets = torch.sum(rewards).unsqueeze(-1) + (gamma * Q_targets_next * (1 - dones[:, i].unsqueeze(-1)))
            # Compute critic loss
            Q_expected = a.critic_network(states, actions)
            # TODO try torch.nn.SmoothL1Loss()
            critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
            # Minimize the loss
            a.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(a.critic_network.parameters(), GRADIENT_CLIPPING_MAX)
            a.optimizer_critic.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actor_loss = -a.critic_network(states, actions_pred).mean()
            # Minimize the loss
            a.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(a.actor_network.parameters(), GRADIENT_CLIPPING_MAX)
            a.optimizer_actor.step()

            self.last_actor_loss[i] = actor_loss.cpu().detach().item()
            self.last_critic_loss[i] = critic_loss.cpu().detach().item()

        for a in self.agents:
            a.soft_update_networks()

    def reset(self):
        for a in self.agents:
            a.reset()

    def save_weights(self, file_name_prefix="weights"):
        for i, a in enumerate(self.agents):
            torch.save(a.actor_network.state_dict(), "{}_agent{}_actor.pth".format(file_name_prefix, i))
            torch.save(a.critic_network.state_dict(), "{}_agent{}_critic.pth".format(file_name_prefix, i))

    def load_weights(self, file_name_prefix="weights"):
        # TODO refactor for code duplication
        for i, a in enumerate(self.agents):
            if self.device.type == "cpu":
                actor_weights_file_name = "{}_agent{}_actor.pth".format(file_name_prefix, i)
                try:
                    a.actor_network.load_state_dict(torch.load(actor_weights_file_name, map_location="cpu"))
                except Exception:
                    print("Could not load actor network weights from {}. Network will be initialized with random "
                        "weights.".format(actor_weights_file_name))
                if a.critic_network:
                    critic_weights_file_name = "{}_agent{}_critic.pth".format(file_name_prefix, i)
                    try:
                        a.critic_network.load_state_dict(torch.load(critic_weights_file_name, map_location="cpu"))
                    except Exception:
                        print("Could not load critic network weights from {}. Network will be initialized with random "
                            "weights.".format(critic_weights_file_name))
            else:
                actor_weights_file_name = "{}_agent{}_actor.pth".format(file_name_prefix, i)
                try:
                    a.actor_network.load_state_dict(torch.load(actor_weights_file_name))
                except Exception:
                    print("Could not load actor network weights from {}. Network will be initialized with random "
                        "weights.".format(actor_weights_file_name))
                if a.critic_network:
                    critic_weights_file_name = "{}_agent{}_critic.pth".format(file_name_prefix, i)
                    try:
                        a.critic_network.load_state_dict(torch.load(critic_weights_file_name))
                    except Exception:
                        print("Could not load critic network weights from {}. Network will be initialized with random "
                            "weights.".format(critic_weights_file_name))

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        random.seed(seed)

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
        assert(len(states) == len(actions) == len(rewards) == len(next_states) == len(dones))
        states_flat = [item for sublist in states for item in sublist]
        actions_flat = [item for sublist in actions for item in sublist]
        next_states_flat = [item for sublist in next_states for item in sublist]
        e = self.experience(states_flat, actions_flat, rewards, next_states_flat, dones)
        self.memory.append(e)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
