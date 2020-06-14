import argparse
import torch
import numpy as np
from statistics import mean
from tensorboardX import SummaryWriter

from agent.a2c_agent import A2CAgent
from agent.ddpg_agent import DDPGAgent
from environment.reacher_environment import ReacherEnvironment

DEFAULT_N_EPISODES_TRAINING = 500
SCORING_WINDOW = 100
PRINT_INTERVAL = 10
DEFAULT_NETWORK_WEIGHTS_FILENAMES = ["actor_weights.pth", "critic_weights.pth"]

class Reacher:
    def __init__(self, agent_type, training_mode=False, headless=False, multiagent=False):
        self.train = training_mode
        self.env = ReacherEnvironment(training_mode=self.train, headless=headless, multiagent=multiagent)
        if agent_type == "a2c":
            self.agent = A2CAgent(self.env.state_size, self.env.action_size, train=self.train, seed=0)
        elif agent_type == "ddpg":
            self.agent = DDPGAgent(self.env.state_size, self.env.action_size, train=self.train, seed=0)
        else:
            raise ValueError("Agent {} is unsupported. Choices are: ddpg, a2c.".format(agent_type))

    def close(self):
        self.env.close()

    def play_episode(self, eps=0.0, beta=1.0):
        states = self.env.reset()
        scores = [0] * self.env.num_agents
        self.agent.reset()

        while True:
            actions = self.agent.act(states)
            next_states, rewards, dones = self.env.step(actions)

            if self.train:
                self.agent.step(states, actions, rewards, next_states, dones)

            scores = [s + r for s, r in zip(scores, rewards)]
            states = next_states

            if any(dones):
                return mean(scores)

    def train_agent(self, num_episodes=DEFAULT_N_EPISODES_TRAINING, output_actor_weights_file="actor_weights.pth",
                    output_critic_weights_file="critic_weights.pth", name=None):
        scores = []
        environment_solved = False

        writer = SummaryWriter("runs/{}".format(name) if name else None)

        for i_episode in range(1, num_episodes + 1):
            episode_score = self.play_episode()

            score = episode_score
            scores.append(score)

            writer.add_scalar("Score", score, i_episode)
            writer.add_scalar("Mean_Score", np.mean(scores[-SCORING_WINDOW:]), i_episode)
            writer.add_scalar("Actor_Loss", self.agent.last_actor_loss, i_episode)
            writer.add_scalar("Critic_Loss", self.agent.last_critic_loss, i_episode)

            if np.mean(scores[-SCORING_WINDOW:]) >= 30 and not environment_solved:
                episode_solved = i_episode - SCORING_WINDOW
                print("Environment solved in {} episodes!".format(episode_solved))
                environment_solved = True
                self.save_weights("{}_solved_e{}".format(output_actor_weights_file, episode_solved),
                                  "{}_solved_e{}".format(output_critic_weights_file, episode_solved))

            if i_episode % PRINT_INTERVAL == 0:
                print("Episode {} -- Average Score: {}".format(i_episode, np.mean(scores[-SCORING_WINDOW:])))

        self.save_weights(output_actor_weights_file, output_critic_weights_file)

    def save_weights(self, actor_weights_file_name, critic_weights_file_name):
        torch.save(self.agent.actor_network.state_dict(), actor_weights_file_name)
        torch.save(self.agent.critic_network.state_dict(), critic_weights_file_name)

    def load_weights(self, actor_weights_file_name, critic_weights_file_name):
        if self.agent.device.type == "cpu":
            self.agent.actor_network.load_state_dict(torch.load(actor_weights_file_name, map_location="cpu"))
            if self.agent.critic_network:
                self.agent.critic_network.load_state_dict(torch.load(critic_weights_file_name, map_location="cpu"))
        else:
            self.agent.actor_network.load_state_dict(torch.load(actor_weights_file_name))
            if self.agent.critic_network:
                self.agent.critic_network.load_state_dict(torch.load(critic_weights_file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Deep Reinforcement Learning agent, or group of agents, which "
                                                 "is a double jointed arm robot with the goal of moving the end of the "
                                                 "into a constantly moving goal area. This is was developed as part of "
                                                 "solving the \"Continuous Control Project\" for the Udacity Deep "
                                                 "Reinforcement Learning Nanodegree.")
    parser.add_argument("--headless", action="store_true", help="Run the application in headless mode, i.e. "
                        "disable the visualization.")
    parser.add_argument("--train", action="store_true", help="Run the application in training mode. This will train "
                        "the agent to solve the environment given. By default, the REINFORCE algorithm is used for "
                        "training.")
    parser.add_argument("--multiagent", action="store_true", help="Run the application with multiple agents (20) in "
                        "in parallel and train the algorithm with more diverse experience.")
    parser.add_argument("-a", "--algorithm", type=str, choices=["a2c", "ddpg"], default="ddpg",
                        help="Algorithm which should be used for training or for inference. Currently only Advantage "
                        "Actor-Critic (A2C) and Deep Deterministic Policy Gradient (DDPG) are implemented.")
    parser.add_argument("-s", "--save_parameters", nargs=2, default=DEFAULT_NETWORK_WEIGHTS_FILENAMES,
                        metavar=("ACTOR_FILENAME", "CRITIC_FILENAME"), help="Path to actor and critic files where the "
                        "parameters from training the agent should be saved to.")
    parser.add_argument("-l", "--load_parameters", nargs=2, default=DEFAULT_NETWORK_WEIGHTS_FILENAMES,
                        metavar=("ACTOR_FILENAME", "CRITIC_FILENAME"), help="Load the agent with the given "
                        "parameters/weights for the actor and critic neural networks.")
    parser.add_argument("--name", type=str, default=None, help="A name for the current training run which will be "
                        "passed to the TensorBoard SummaryWriter.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_N_EPISODES_TRAINING,
                        help="Number of episodes to run during training. Default = 500.")
    args = parser.parse_args()
    
    reacher = Reacher(agent_type=args.algorithm, training_mode=args.train, headless=args.headless,
                      multiagent=args.multiagent)

    if args.train:
        reacher.train_agent(num_episodes=args.episodes, output_actor_weights_file=args.save_parameters[0],
                            output_critic_weights_file=args.save_parameters[1], name=args.name)
    else:
        reacher.load_weights(actor_weights_file_name=args.load_parameters[0],
                             critic_weights_file_name=args.load_parameters[1])
        score = reacher.play_episode()
        print("Score: {.2f}".format(score))

    reacher.close()
