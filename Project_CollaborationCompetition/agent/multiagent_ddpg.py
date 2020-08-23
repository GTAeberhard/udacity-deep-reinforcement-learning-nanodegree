"""
Implementation of the multi-agent DDPG algorithm.
"""
import random
import numpy as np
from collections import namedtuple, deque

import torch
import torch.nn.functional as F

from .hyperparameters import DEFAULT_HYPERPARAMETERS
from .ddpg_agent import DDPGAgent


class MultiAgentDDPG:
    """
    Object which implements the multi-agent version of the Deep Deterministic Policy Gradient algorithm using only a
    single, shared actor and single critic neural network.

    Attributes
    ----------
    state_size: Input state space size for the agents' environment
    action_size: Output action space size for the agents' environment
    num_agents: Number of agents in the environment - used solely for determining size of critic network input size
    train: True if agent is configured to be in training mode
    agents: Class which represents a single DDPG agent, where in this case the single DDPG agent is shared
    memory: Replay experience buffer
    steps: Counter which keeps track of the number of steps which have been executed by the agent
    episodes: Counter which keeps track of the number of episodes which the agent has experienced
    last_actor_loss: The last loss value from the last training optimization iteration of the actor network
    last_critic_loss: The last loss value from the last training optimization iteration of the critic network
    hyperparameters: Dictionary which contains all of the necessary hyperparameters
    eps: Runing value of epsilon which is used for scaling the amount of noise added to the actions in order to balance
        exploration and exploitation
    device: PyTorch device where the data and calculations reside
    """
    def __init__(self, state_size, action_size, num_agents, train=False, hyperparameters=DEFAULT_HYPERPARAMETERS,
                 seed=0, device=None):
        """
        Initializes the multi-agent DDPG algorithm implementation, including the initialization of the required neural
        networks.

        Parameters
        ----------
        state_size: Input state space size for the agents' environment
        action_size: Output action space size for the agents' environment
        num_agents: Number of agents in the environment - used solely for determining size of critic network input size
        train: True if agent is configured to be in training mode
        hyperparameters: Dictionary which contains all of the necessary hyperparameters
        seed: Random seed number
        device: PyTorch device where the data and calculations reside
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.train = train
        self.steps = 0
        self.episodes = 0
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.hyperparameters = hyperparameters
        self.eps = self.hyperparameters["EPSILON_START"]

        np.random.seed(seed)
        torch.manual_seed(seed)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.agents = DDPGAgent(self.state_size, self.action_size, self.num_agents, self.train, self.hyperparameters,
                                seed, device)

        if self.train:
            self.memory = ReplayBuffer(self.hyperparameters["BUFFER_SIZE"], self.hyperparameters["BATCH_SIZE"], seed,
                                       self.device)

    def act(self, states_input):
        """
        Have all of the agents in the environment produce actions given the environment's current state. With the
        hyperparameter WARM_UP_EPISODES, an initial amount of episodes can be run using a random actions sampled from a
        uniform distribution, in order to fill the replay experience buffer with an initial variety of experiences.

        Have the agent produce an action in the environment given the environment's current state. Uses the actor
        network to produce an output action vector. Gradients are not calculated while doing the forward pass through
        the actor network.

        Parameters
        ----------
        states_input: Current local input state of the environment for all of the agents

        Returns
        -------
        actions: Output actions that all of the agents should take in the environment
        """
        if self.episodes < self.hyperparameters["WARM_UP_EPISODES"] and self.train:
            # Take a random action during the warm up phase to gather experience
            return np.random.uniform(
                low=-1, high=1, size=self.action_size * self.num_agents
            ).reshape(self.num_agents, self.action_size)
        else:
            actions_agent = torch.empty(self.num_agents, self.action_size).to(self.device)
            for i in range(self.num_agents):
                states_agent = torch.from_numpy(states_input[i]).float().unsqueeze(0).to(self.device)
                actions_agent[i, :] = self.agents.act(states_agent, self.eps)

            return actions_agent.squeeze(0).cpu().numpy()

    def step(self, states, actions, rewards, next_states, dones):
        """
        Take a training step for the multi-agent DDPG algorithm. For every step, it adds the experience to the
        replay experience buffer and calls the method for executing a training step.

        Parameters
        ----------
        states: Current state of the environment for all of the agents
        actions: Actions taken by all of the agents given the current state
        rewards: Rewards received by all of the agents from the environment, given 'states' and taken 'actions'
        next_states: The next state of the environment for all of the agents, after 'actions' were taken
        dones: Boolean flags which indicate for each agent of the end of the episode has been reached
        """
        self.steps += 1
        if self.train:
            self.memory.add(states, actions, rewards, next_states, dones)
            if self.episodes >= self.hyperparameters["WARM_UP_EPISODES"]:
                self.learn(self.hyperparameters["GAMMA"])
                self.eps = max(self.eps - self.hyperparameters["EPSILON_DECAY"], self.hyperparameters["EPSILON_END"])
            if np.any(dones):
                self.episodes += 1

    def learn(self, gamma=DEFAULT_HYPERPARAMETERS["GAMMA"]):
        """
        Update policy (actor) and value (critic) parameters using a set of experience tuples. Update gets executed twice
        on the same actor and critic network, once for each agent. Reward is chosen as the sum of the rewards across all
        agents in order to encourage agents to cooperate, i.e. achieve the highest score possible together.

        Parameters
        ----------
        gamma: Discount factor
        """
        if not (len(self.memory) > self.hyperparameters["BATCH_SIZE"] and 
                self.steps % self.hyperparameters["STEPS_BETWEEN_LEARNING"] == 0):
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Pre-calculate next actions (from target network) and predicted actions (with current local network) for all
        # agents
        for i in range(self.num_agents):
            actions_next = actions.clone()
            actions_pred = actions.clone()
            for j in range(self.num_agents):
                i_start_states = j * self.state_size
                i_end_states = i_start_states + self.state_size
                agent_states = states[:, i_start_states:i_end_states]
                agent_next_states = next_states[:, i_start_states:i_end_states]

                i_start_actions = j * self.action_size
                i_end_actions = i_start_actions + self.action_size
                actions_next[:, i_start_actions:i_end_actions] = self.agents.target_actor_network(agent_next_states)
                actions_pred[:, i_start_actions:i_end_actions] = self.agents.actor_network(agent_states)

            # Reward for each learning iteration is the sum of the rewards for both agents, in order to encourage the
            # agents to collaborate and achieve the highest possible score in the environment together.
            # Motivated from implementation of https://github.com/shartjen/Tennis-MADDPG/blob/master/ddpg_agent.py
            agent_reward = torch.sum(rewards).unsqueeze(-1)

            agent_dones = dones[:, i].unsqueeze(-1)
            
            self.last_actor_loss, self.last_critic_loss = self.agents.learn(
                states, actions, next_states, actions_next, actions_pred, agent_reward, agent_dones, gamma
            )

        self.agents.soft_update_networks(self.hyperparameters["TAU"])

    def save_weights(self, file_name_actor="weights_actor.pth", file_name_critic="weights_critic.pth"):
        """Save the current actor and critic neural network weights to a file."""
        torch.save(self.agents.actor_network.state_dict(), file_name_actor)
        torch.save(self.agents.critic_network.state_dict(), file_name_critic)

    def load_weights(self, file_name_actor="weights_actor.pth", file_name_critic="weights_critic.pth"):
        """Load actor and critic network weights from a file."""
        try:
            self.load_single_network_weights(self.agents.actor_network, file_name_actor)
        except Exception:
            print("Could not load actor network weights from {}. Network will be initialized with random "
                  "weights.".format(file_name_actor))
        if self.agents.critic_network:
            try:
                self.load_single_network_weights(self.agents.critic_network, file_name_critic)
            except Exception:
                print("Could not load critic network weights from {}. Network will be initialized with random "
                    "weights.".format(file_name_critic))

    def load_single_network_weights(self, network, file_name):
        """Load a given network's weights from a file."""
        if self.device.type == "cpu":
            network.load_state_dict(torch.load(file_name, map_location="cpu"))
        else:
            network.load_state_dict(torch.load(file_name))
   
class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size, seed, device):
        """
        Initialize a ReplayBuffer object.
        
        Parameters
        ----------
        buffer_size: maximum size of buffer
        batch_size: size of each training batch
        seed: random seed used during sampling
        device: PyTorch device where the data should be loaded
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
