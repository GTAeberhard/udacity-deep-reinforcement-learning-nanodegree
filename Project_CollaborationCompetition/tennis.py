import argparse
import torch
import numpy as np

from agent.human_agents import HumanAgents
from environment.tennis_environment import TennisEnvironment


class Tennis:
    def __init__(self, mode="inference", headless=False, device=None):
        assert(mode == "inference" or mode == "train" or mode == "manual")

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
            raise ValueError("Currently only manual mode is supported.")

    def play_episode(self):
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

    def close(self):
        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Deep Reinforcement Learning with two agents which play a game of "
                                                 "tennis against one another. This was developed as part of solving the "
                                                 "\"Collaboration and Competition Project\" for the Udacity Deep "
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
    args = parser.parse_args()

    if args.mode == "manual" and args.headless:
        raise RuntimeError("Cannot run manual and headless mode simultaneously (how else are you suppose to see "
                           "anything?).")

    tennis = Tennis(mode=args.mode, headless=args.headless)

    if args.mode == "train":
        raise RuntimeError("Training not yet implemented.")
    else:
        score = tennis.play_episode()
        print("Score: {}".format(score))

    tennis.close()
    