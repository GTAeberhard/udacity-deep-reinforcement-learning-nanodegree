import argparse
import numpy as np

from agent.agent import Agent
from environment.reacher_environment import ReacherEnvironment

DEFAULT_N_EPISODES_TRAINING = 1000
SCORING_WINDOW = 100

class Reacher:
    def __init__(self, training_mode=False, headless=False):
        self.env = ReacherEnvironment(training_mode=False, headless=headless)
        self.agent = Agent(self.env.state_size, self.env.action_size, seed=0)

    def close(self):
        self.env.close()

    def play_episode(self, eps=0.0, beta=1.0):
        state = self.env.reset()
        score = 0

        while True:
            actions = self.agent.act(state)
            next_state, reward, done = self.env.step(actions)

            score += reward
            state = next_state

            if done:
                return score

    def train_agent(self, num_episodes=DEFAULT_N_EPISODES_TRAINING):
        scores = []
        environment_solved = False

        for i_episode in range(1, num_episodes + 1):
            score = self.play_episode()

            scores.append(score)

            # writer.add_scalar("Score", score, i_episode)
            # writer.add_scalar("Mean_Score", np.mean(scores[-SCORING_WINDOW:]), i_episode)

            if np.mean(scores[-SCORING_WINDOW:]) >= 30 and not environment_solved:
                print("Environment solved in {} episodes!".format(i_episode))
                environment_solved = True

            if i_episode % SCORING_WINDOW == 0:
                print("Episode {} -- Average Score: {}".format(i_episode, np.mean(scores[-SCORING_WINDOW:])))



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
    args = parser.parse_args()
    
    reacher = Reacher(training_mode=args.train, headless=args.headless)

    if args.train:
        reacher.train_agent()
    else:
        score = reacher.play_episode()
        print("Score: {}".format(score))

    reacher.close()
