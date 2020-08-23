"""
Implementation of a single DDPG agent.
"""
import copy
import random
import numpy as np

import torch
import torch.nn.functional as F

from .hyperparameters import DEFAULT_HYPERPARAMETERS
from .network import DDPGActorNetwork, MultiAgentDDPGCriticNetwork

class DDPGAgent:
    """
    Actor-Critic Deep Deterministic Policy Gradient (DDPG) agent implementation for a single agent, with the possibility
    of specifying several agents for the purpose of having a multi-agent critic network.

    Attributes
    ----------
    state_size: Input state space size for the agent's environment
    action_size: Output action space size for the agent's environment
    num_agents: Number of agents in the environment - used solely for determining size of critic network input size
    train: True if agent is configured to be in training mode
    actor_network: Actor network used for inference and training
    critic_network: Critic network used for training
    optimizer_actor: Optimizer used for the training the actor network
    optimizer_critic: Optimizer used for training the critic network
    target_actor_network: Target actor network used during training
    target_critic_network: Target critic network used during training
    device: PyTorch device where the data and calculations reside
    """
    def __init__(self, state_size, action_size, num_agents, train=False, hyperparameters=DEFAULT_HYPERPARAMETERS,
                 seed=0, device=None):
        """
        Initialize the DDPG agent with all necessary attributes required for inference and/or training, such as
        creating the neural networks.

        Parameters
        ----------
        state_size: Input state space size for the agent's environment
        action_size: Output action space size for the agent's environment
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
        self.hyperparameters = hyperparameters

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.actor_network = DDPGActorNetwork(
            self.state_size, self.action_size, seed, self.hyperparameters["DNN_ARCHITECTURE"]
        ).to(self.device)

        if self.train:
            self.critic_network = MultiAgentDDPGCriticNetwork(
                self.state_size, self.action_size, self.num_agents, seed, self.hyperparameters["DNN_ARCHITECTURE"]
            ).to(self.device)

            self.optimizer_actor = torch.optim.Adam(self.actor_network.parameters(),
                                                    lr=self.hyperparameters["LEARNING_RATE_ACTOR"])
            self.optimizer_critic = torch.optim.Adam(self.critic_network.parameters(),
                                                     lr=self.hyperparameters["LEARNING_RATE_CRITIC"],
                                                     weight_decay=self.hyperparameters["WEIGHT_DECAY"])

            self.actor_network.train()
            self.critic_network.train()

            self.target_actor_network = DDPGActorNetwork(self.state_size, self.action_size).to(self.device)
            self.target_critic_network = MultiAgentDDPGCriticNetwork(
                self.state_size, self.action_size, self.num_agents
            ).to(self.device)

            self._hard_update(self.actor_network, self.target_actor_network)
            self._hard_update(self.critic_network, self.target_critic_network)

            self.noise = OUNoise(self.action_size, 0)
        else:
            self.actor_network.eval()
            self.critic_network = None

    def act(self, states, eps=1.0):
        """
        Have the agent produce an action in the environment given the environment's current state. Uses the actor
        network to produce an output action vector. Gradients are not calculated while doing the forward pass through
        the actor network. If in training mode, noise is added to the actions in order to encourage exploration of the
        state space.

        Parameters
        ----------
        states: Current input state of the environment for the agent
        eps: Scaling factoring for the additive noise - usually used to scale down exploration during training

        Returns
        -------
        actions: Output actions that the agent should take in the environment
        """
        self.actor_network.eval()
        with torch.no_grad():
            actions = self.actor_network(states)
            if self.train:
                noise = torch.from_numpy(
                    self.hyperparameters["NOISE_SIGMA"] * np.random.rand(self.action_size)
                ).float().unsqueeze(0).to(self.device)
                actions += eps * noise
            actions.clamp_(-1, 1).squeeze(0).cpu().numpy()
        if self.train:
            self.actor_network.train()

        return actions

    def learn(self, states, actions, next_states, next_actions, actions_predicted, rewards, dones, gamma):
        """
        Calculate a single training pass for the actor and critic networks for this agent. Note that all inputs could
        be stacked in the case of multiple agents.

        Parameters
        ----------
        states: Current states of the environment
        actions: Actions taken at the current states
        next_states: The next states in the environment, given that the actions in 'actions' were taken
        next_actions: The next actions to be taken, as calculated by a target actor network given the states from
            'next_states'
        actions_predicted: The predicted actions to be taken, as calculated by the actor network given the current state
        rewards: Rewards received by the environment when 'actions' were taken from 'states'
        dones: Boolean which signals if the end of an episode was reached
        gamma: Discount factor

        Returns
        -------
        actor_loss: Loss value during the training pass for the actor network
        critic_loss: Loss value during the training pass for the critic network
        """
        critic_loss = self._update_critic(states, actions, next_states, next_actions, rewards, dones, gamma)
        actor_loss = self._update_actor(states, actions_predicted)
        return actor_loss, critic_loss

    def _update_critic(self, states, actions, next_states, next_actions, rewards, dones, gamma):
        """
        Update the critic network using a TD estimate w.r.t. a target critic network

        Parameters
        ----------
        states: Current states of the environment
        actions: Actions taken at the current states
        next_states: The next states in the environment, given that the actions in 'actions' were taken
        next_actions: The next actions to be taken, as calculated by a target actor network given the states from
            'next_states'
        rewards: Rewards received by the environment when 'actions' were taken from 'states'
        dones: Boolean which signals if the end of an episode was reached
        gamma: Discount factor
        
        Returns
        -------
        critic_loss: Loss value during the training pass for the critic network
        """
        with torch.no_grad():
            Q_targets_next = self.target_critic_network(next_states, next_actions)
        
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.critic_network(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.hyperparameters["GRADIENT_CLIPPING_MAX"])
        self.optimizer_critic.step()

        return critic_loss.cpu().detach().item()

    def _update_actor(self, states, actions_predicted):
        """
        Update the actor network with gradients from the critic network, calculated using the current state and
        predicted actions which would be taken with the current actor network.

        Parameters
        ----------
        states: Current states of the environment
        actions_predicted: The predicted actions to be taken, as calculated by the actor network given the current state

        Returns
        -------
        actor_loss: Loss value during the training pass for the actor network
        """
        actor_loss = -self.critic_network(states, actions_predicted).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.hyperparameters["GRADIENT_CLIPPING_MAX"])
        self.optimizer_actor.step()

        return actor_loss.cpu().detach().item()

    def soft_update_networks(self, tau=DEFAULT_HYPERPARAMETERS["TAU"]):
        """Execute soft update of the agent's actor and critic networks"""
        self._soft_update(self.actor_network, self.target_actor_network, tau)
        self._soft_update(self.critic_network, self.target_critic_network, tau)

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        """
        Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau: interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    @staticmethod
    def _hard_update(local_model, target_model):
        """
        Hard update of the model parameters, i.e. directly copy the parameters from model to another

        Parameters
        ----------
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
