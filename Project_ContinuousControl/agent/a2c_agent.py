import random
import numpy as np

import torch
from torch.distributions import Normal

from .hyperparameters import LEARNING_RATE, GAMMA, BELLMAN_STEPS, ENTROPY_COEFFICIENT, GRADIENT_CLIPPING_MAX
from .network import A2CActorNetwork, A2CCriticNetwork, A2CActorCriticNetwork


class A2CAgent():
    """An agent that interacts with and learns from the environment using policy-based DRL methods."""

    def __init__(self, state_size, action_size, train=False, seed=0, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.train = train
        self.last_actor_loss = None
        self.last_critic_loss = None
        self.actor_grads = None
        self.critic_grads = None

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.actor_network = A2CActorNetwork(self.state_size, self.action_size).to(self.device)
        self.critic_network = A2CCriticNetwork(self.state_size).to(self.device)

        self.actor_critic_network = A2CActorCriticNetwork(self.state_size, self.action_size).to(self.device)

        if self.train:
            self.optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=LEARNING_RATE)
            self.optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=LEARNING_RATE)
            self.actor_network.train()
            self.critic_network.train()
            self.reset()
        else:
            self.actor_network.eval()
            self.critic_network.eval()

    def act(self, states):
        actions = np.empty([len(states), self.action_size])
        log_probs = torch.empty(len(states), 1).to(self.device)
        v_values = torch.empty(len(states), 1).to(self.device)
        entropies = torch.empty(len(states), 1).to(self.device)
        for i in range(len(states)):
            state = torch.from_numpy(states[i]).float().unsqueeze(0).to(self.device)
            mean, sigma = self.actor_network(state)
            v_value = self.critic_network(state)
            actions_distribution = Normal(mean, sigma)
            selected_actions = actions_distribution.sample().clamp(-1, 1)
            actions[i, :] = selected_actions
            log_probs[i, :] = actions_distribution.log_prob(selected_actions).sum(-1).unsqueeze(-1)
            entropies[i, :] = actions_distribution.entropy().sum(-1).unsqueeze(-1)
            v_values[i, :] = v_value

        if self.train:
            if self.trajectory_log_probs is None:
                self.trajectory_log_probs = log_probs.unsqueeze(2)
                self.trajectory_v_values = v_values.unsqueeze(2)
                self.trajectory_entropies = entropies.unsqueeze(2)
            else:
                self.trajectory_log_probs = torch.cat([self.trajectory_log_probs, log_probs.unsqueeze(2)], dim=2)
                self.trajectory_v_values = torch.cat([self.trajectory_v_values, v_values.unsqueeze(2)], dim=2)
                self.trajectory_entropies = torch.cat([self.trajectory_entropies, entropies.unsqueeze(2)], dim=2)

        return actions

    def step(self, states, actions, rewards, next_states, dones):
        if self.trajectory_rewards is None:
            self.trajectory_rewards = np.array([rewards], dtype=float).T
        else:
            self.trajectory_rewards = np.hstack([self.trajectory_rewards, np.array([rewards], dtype=float).T])

        if self.trajectory_rewards.shape[1] == BELLMAN_STEPS:
            final_v_values = torch.empty(len(next_states), 1).to(self.device)
            with torch.no_grad():
                for i in range(len(next_states)):
                    next_state = torch.from_numpy(next_states[i]).float().unsqueeze(0).to(self.device)
                    final_v_values[i, :] = self.critic_network(next_state)
            self.learn(final_v_values)

    def learn(self, final_values=None, gamma=GAMMA):
        assert(len(self.trajectory_log_probs) == len(self.trajectory_rewards) == len(self.trajectory_v_values))

        if final_values is not None:
            final_values = final_values.detach().squeeze(1).cpu().numpy()

        discounted_rewards = torch.Tensor(
            self._calculate_discounted_rewards(self.trajectory_rewards, final_values)
        ).to(self.device).unsqueeze(1).reshape(-1)

        self.trajectory_v_values = self.trajectory_v_values.reshape(-1)

        advantage = discounted_rewards - self.trajectory_v_values

        self.trajectory_log_probs = self.trajectory_log_probs.reshape(-1)
        self.trajectory_entropies = self.trajectory_entropies.reshape(-1)

        entropy_loss = ENTROPY_COEFFICIENT * self.trajectory_entropies.mean()
        policy_loss = (-advantage.detach() * self.trajectory_log_probs).mean()
        policy_entropy_loss = policy_loss - entropy_loss
        value_loss = 0.5 * advantage.pow(2).mean()
        
        self.optimizer_actor.zero_grad()
        policy_entropy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), GRADIENT_CLIPPING_MAX)
        self.actor_grads = np.concatenate(
            [p.grad.data.cpu().numpy().flatten() for p in self.actor_network.parameters() if p.grad is not None]
        )
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), GRADIENT_CLIPPING_MAX)
        self.critic_grads = np.concatenate(
            [p.grad.data.cpu().numpy().flatten() for p in self.critic_network.parameters() if p.grad is not None]
        )
        self.optimizer_critic.step()

        self.reset()

        self.last_actor_loss = policy_loss.item()
        self.last_critic_loss = value_loss.item()

    def reset(self):
        self.trajectory_log_probs = None
        self.trajectory_rewards = None
        self.trajectory_v_values = None
        self.trajectory_entropies = None

    @staticmethod
    def _calculate_discounted_rewards(rewards_list, final_values=None, gamma=GAMMA):
        if final_values is None:
            final_values = [0] * len(rewards_list)
        assert(len(rewards_list) == len(final_values))
        # Implementation from Lapan, "Deep Reinforcement Learning Hands-On", page 246
        discounted_rewards_list = []
        for rewards, final_value in zip(rewards_list, final_values):
            discounted_rewards = []
            sum_rewards = final_value
            for r in reversed(rewards):
                sum_rewards *= gamma
                sum_rewards += r
                discounted_rewards.append(sum_rewards)
            discounted_rewards = list(reversed(discounted_rewards))
            discounted_rewards_list.append(discounted_rewards)

        return discounted_rewards_list
