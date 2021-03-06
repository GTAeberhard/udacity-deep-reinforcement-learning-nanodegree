"""
Main application for running the Tennis environment for inference or for training. A manual
mode using the keyboard is also supported.
"""
import argparse
import json
import torch
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

from agent.human_agents import HumanAgents
from agent.multiagent_ddpg import MultiAgentDDPG
from agent.hyperparameters import DEFAULT_HYPERPARAMETERS
from environment.tennis_environment import TennisEnvironment

DEFAULT_N_EPISODES_TRAINING = 3000
DEFAULT_N_EPISODES_MANUAL = 5
SCORING_WINDOW = 100
PRINT_INTERVAL = 10
DEFAULT_NETWORK_WEIGHTS_FILENAMES = ["weights_actor.pth", "weights_critic.pth"]
DEFAULT_RANDOM_SEED = 0


class Tennis:
    """
    Main functional class which coordinates all activies for the Tennis game environment. Three modes for the
    environment are support: inference, training and manual. In inference mode, neural networks are used to control the
    actions of the two agents, where the weights of the networks are pre-loaded from a file. In training mode, the
    environment is run for many episodes with the goal of training new neural networks for the agents. In manual mode,
    the keyboard can be used so that a human user can have a go at playing the tennis game.

    Attributes
    ----------
    env: Tennis environment class which contains the high-level API for interacting with the environment
    training_mode: True if training mode is enabled
    device: PyTorch device variable defining on which device the data should be loaded
    agents: Multi agent class which contains the logic for the agents acting in the environment
    """
    def __init__(self, mode="inference", headless=False, hyperparameters=DEFAULT_HYPERPARAMETERS, seed=0, device=None):
        """
        Initializes all of the attributes of the Tennis environment.

        Parameters
        ----------
        mode: The mode in which to run the Tennis game (inference, training or manual mode)
        headless: Set to True if the environment should be run in headless mode, i.e. with graphical output
        hyperparameters: Dictionary which contains all of the necessary hyperparameters for the multi-agent DDPG
        seed: Random seed number
        device: Device on which to run the agents (cpu or cuda)
        """
        assert(mode == "inference" or mode == "train" or mode == "manual"), \
            "Unsupported mode {}. Possible options: inference, train, manual".format(mode)

        self.training_mode = True if mode == "train" else False

        self.env = TennisEnvironment(training_mode=self.training_mode, headless=headless)

        if device == "cuda":
            self.device = torch.device("cuda:0")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if mode == "manual":
            self.agents = HumanAgents()
        else:
            self.agents = MultiAgentDDPG(state_size=self.env.state_size, action_size=self.env.action_size,
                                         num_agents=self.env.num_agents, train=self.training_mode,
                                         hyperparameters=hyperparameters, seed=seed, device=self.device)

    def close(self):
        """Properly close the environment."""
        self.env.close()

    def play_episode(self):
        """
        Play a single episode of the environment until either the ball hits the ground or the ball goes out of bounds.

        Returns
        -------
        scores: Score for each agent at the conclusion of the episode
        """
        states = self.env.reset()
        scores = np.zeros(self.env.num_agents)

        while True:
            actions = self.agents.act(states)
            next_states, rewards, dones = self.env.step(actions)

            if self.training_mode:
                self.agents.step(states, actions, rewards, next_states, dones)

            scores += rewards
            states = next_states

            if np.any(dones):
                return scores

    def train_agent(self, num_episodes=DEFAULT_N_EPISODES_TRAINING, filename_weights_prefix="", name=None):
        """
        Train the agents in the Tennis environment. Training progress is logged to a TensorBoard, as well
        as regularly output to the command line. The weights of the neural networks for the episode which achieves the
        maximum score are automatically saved to a file. The weights are also automatically saved for the episode
        which first solved the environment, i.e. getting an average score over 100 episodes of at least 0.5.

        Parameters
        ----------
        num_episodes: Number of episodes to run the training
        filename_weights_prefix: Prefix which is prepended to the default file names when the network weights are saved
        name: Name of the TensorBoard run
        """
        scores = []
        max_score = 0
        max_mean_score = 0
        environment_solved = False

        filename_weights_prefix = "{}_".format(filename_weights_prefix) if filename_weights_prefix else ""

        writer = SummaryWriter("runs/{}".format(name) if name else None)

        for i_episode in range(1, num_episodes + 1):
            score = self.play_episode()

            episode_max_score = np.max(score)
            scores.append(episode_max_score)

            mean_score = np.mean(scores[-SCORING_WINDOW:])
            writer.add_scalar("Score", episode_max_score, i_episode)
            writer.add_scalar("Mean_Score", mean_score, i_episode)
            if self.agents.last_actor_loss:
                writer.add_scalar("Actor_Loss", self.agents.last_actor_loss, i_episode)
            if self.agents.last_critic_loss:
                writer.add_scalar("Critic_Loss", self.agents.last_critic_loss, i_episode)

            if episode_max_score > max_score:
                max_score = episode_max_score

            if mean_score > max_mean_score:
                max_mean_score = mean_score
                self.agents.save_weights(file_name_actor="{}weights_max_actor.pth".format(filename_weights_prefix),
                                         file_name_critic="{}weights_max_critic.pth".format(filename_weights_prefix))

            if mean_score >= 0.5 and not environment_solved:
                print("Environment solved in {} episodes!".format(i_episode))
                environment_solved = True
                self.agents.save_weights(
                    file_name_actor="{}weights_solved_e{}_actor.pth".format(filename_weights_prefix, i_episode),
                    file_name_critic="{}weights_solved_e{}_critic.pth".format(filename_weights_prefix, i_episode)
                )

            if i_episode % PRINT_INTERVAL == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("{}: Episode {} -- Average Score: {:.5f} -- Max Average Score: {:.5f} -- Max Score: {:.5f} -- "
                      "Eps: {:.5f}".format(current_time, i_episode, mean_score, max_mean_score, max_score,
                                           self.agents.eps))

    def load_weights(self, file_name_actor="weights_actor.pth", file_name_critic="weights_critic.pth"):
        """Load actor and critic network weights from a file."""
        self.agents.load_weights(file_name_actor, file_name_critic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Deep Reinforcement Learning with two agents which play a game of "
                                                 "tennis against one another. This was developed as part of solving the"
                                                 " \"Collaboration and Competition Project\" for the Udacity Deep "
                                                 "Reinforcement Learning Nanodegree.")
    parser.add_argument("-m", "--mode", type=str, choices=["inference", "train", "manual"], default="inference",
                        help="The application can be run in inference, training and manual mode. In inference mode,"
                             "a multi-agent deep reinforcement learning algorithm is used to run a single episode of "
                             "the environment. In training mode, several episodes will be run to train a the agents. "
                             "In manual mode, the keyboard can be used to control the agents manually with the "
                             "folling commands (left agent, right agent): (w, up-arrow) Up, (s, down-arrow) Down, (a, "
                             "right-arrow) Right, (d, left-arrow) Left.")
    parser.add_argument("--headless", action="store_true", help="Run the application in headless mode, i.e. "
                        "disable the visualization. This option will not work with manual mode.")
    parser.add_argument("-e", "--episodes", type=int,
                        help="Number of episodes to run for either training or inference.")
    parser.add_argument("-l", "--load_parameters", nargs=2, default=DEFAULT_NETWORK_WEIGHTS_FILENAMES,
                        metavar=("ACTOR_FILENAME", "CRITIC_FILENAME"), help="Load the agent with the given "
                        "parameters/weights for the actor and critic neural networks.")
    parser.add_argument("-s", "--save_parameters_prefix", type=str, default="",
                        help="Prefix to be added to the saved file names for the trained network weights.")    
    parser.add_argument("-p", "--hyperparameters_json", type=str, help="JSON file which contains the values of all of "
                        "the hyperparameters required by the multi-agent DDPG algorithm.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help="Random number seed to be used for all internal calculations")
    args = parser.parse_args()

    if args.mode == "manual" and args.headless:
        raise RuntimeError("Cannot run manual and headless mode simultaneously (how else are you suppose to see "
                           "anything?).")

    if args.hyperparameters_json:
        with open(args.hyperparameters_json) as json_file:
            hyperparameters = json.load(json_file)
    else:
        hyperparameters = DEFAULT_HYPERPARAMETERS

    tennis = Tennis(mode=args.mode, headless=args.headless, hyperparameters=hyperparameters, seed=args.seed)

    if args.mode == "train":
        args.episodes = DEFAULT_N_EPISODES_TRAINING if args.episodes is None else args.episodes
        tennis.train_agent(num_episodes=args.episodes)
    else:
        if args.episodes is None:
            args.episodes = DEFAULT_N_EPISODES_MANUAL
        if args.mode == "inference":
            tennis.load_weights(file_name_actor=args.load_parameters[0], file_name_critic=args.load_parameters[1])
        for i_episode in range(args.episodes):
            score = tennis.play_episode()
            print("Game {}:\n\tPlayer 1: {:.2f}\n\tPlayer 2: {:.2f}".format(i_episode, score[0], score[1]))

    tennis.close()
    