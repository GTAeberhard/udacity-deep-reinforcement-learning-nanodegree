import argparse
import torch
import numpy as np
from collections import deque
from tensorboardX import SummaryWriter

from environment.banana_environmet import BananaEnvironment
from agent.dqn_agent import DqnAgent
from agent.human_agent import HumanAgent


N_EPISODES_TRAINING = 1000
SCORING_WINDOW = 100


class BananaNavigation:
    def __init__(self, mode="inference", double_dqn=False, headless=False):
        assert(mode == "inference" or mode == "train" or mode == "manual")

        self.training_mode = True if mode == "train" else False

        self.env = BananaEnvironment(training_mode=self.training_mode, headless=headless)

        if mode == "manual":
            self.agent = HumanAgent()
        else:
            self.agent = DqnAgent(state_size=self.env.state_size, action_size=self.env.action_size, seed=0,
                                  double_dqn=double_dqn)

    def load_weights(self, weights_file="weights.pth"):
        self.agent.qnetwork_local.load_state_dict(torch.load(weights_file))

    def play_episode(self):
        state = self.env.reset()
        score = 0

        while True:
            action = self.agent.act(state)
            next_state, reward, done = self.env.step(action)

            if self.training_mode:
                self.agent.step(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                return score

    def train_agent(self, num_episodes=N_EPISODES_TRAINING, output_weights_file="weights.pth"):
        scores = []
        eps = 1.0

        writer = SummaryWriter()

        for i_episode in range(1, num_episodes + 1):
            score = self.play_episode()

            scores.append(score)

            eps = max(0.01, 0.995*eps)

            writer.add_scalar("Score", score, i_episode)
            writer.add_scalar("Mean_Score", np.mean(scores[-SCORING_WINDOW:]), i_episode)

            if np.mean(scores[-SCORING_WINDOW:]) >= 13:
                print("Environment solved in {} episodes!".format(i_episode))
                torch.save(self.agent.qnetwork_local.state_dict(), output_weights_file)
                break

            if i_episode % SCORING_WINDOW == 0:
                print("Episode {} -- Average Score: {}".format(i_episode, np.mean(scores[-SCORING_WINDOW:])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Deep Reinforcement Learning agent which navigates the "
                                                 "Banana environment, where the goal is to collect as many yellow "
                                                 "bananas as possible while avoiding the blue bananas. This is was "
                                                 "developed as part of solving the \"Navigation Project\" for the "
                                                 "Udacity Deep Reinforcement Learning Nanodegree.")
    parser.add_argument("-m", "--mode", type=str, choices=["inference", "train", "manual"], default="inference",
                        help="The application can be run in inference, training and manual mode. In inference mode,"
                             "a Deep Q-Network will be used to run a single episode of the environment. In training "
                             "mode, several episodes will be run to train a new Deep Q-Network and save its resulting "
                             "weights. In manual mode, the keyboard can be used to control the agent manually with the "
                             "folling commands: (w) Up, (s) Down, (a) Right, (d) Left.")
    parser.add_argument("-l", "--load_parameters", help="Load the agent with the given parameters/weights for the "
                        "neural network.")
    parser.add_argument("-o", "--options", nargs="*", help="Specfiy additional options which extend the basic DQN "
                        "algorithm. Possible options: double: Double DQN")
    parser.add_argument("--headless", action="store_true", help="Run the application in headless mode, i.e. "
                        "disable the visualization. This option will not work with manual mode.")
    args = parser.parse_args()

    if args.mode == "manual" and args.headless:
        raise RuntimeError("Cannot run the manual and headless mode simultaneously (how else are you suppose to see "
                           "anything?).")

    double_dqn = True if args.options is not None and "double" in args.options else False

    banana_navigation = BananaNavigation(mode=args.mode, double_dqn=double_dqn, headless=args.headless)

    if not args.mode == "manual" and args.load_parameters:
        banana_navigation.load_weights(args.load_parameters)

    if args.mode == "train":
        banana_navigation.train_agent(num_episodes=N_EPISODES_TRAINING)
    else:
        score = banana_navigation.play_episode()
        if score >= 15:
            print("\nWow, you are a banana-collecting legend!\n\nScore: {}".format(score))
        elif score >= 10:
            print("\nWell done! That's enough bananas for the whole monkey family!\n\nScore: {}".format(score))
        elif score >= 5:
            print("\nOk, so you were able to collect a few bananas, not bad, but I think you can do better!"
                  "\n\nScore: {}".format(score))
        else:
            print("\nEverything OK? Why aren't you moving? You know you should be going after the YELLOW bananas, "
                  "right? I am a bit disappointed in you...\n\nScore: {}".format(score))
