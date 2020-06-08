import argparse
import numpy as np

from environment.reacher_environment import ReacherEnvironment

class Reacher:
    def __init__(self, headless=False):
        self.env = ReacherEnvironment(headless=headless)

    def close(self):
        self.env.close()

    def play_episode(self, eps=0.0, beta=1.0):
        state = self.env.reset()
        score = 0

        while True:
            action = np.random.uniform(low=-1, high=1, size=self.env.action_size)
            next_state, reward, done = self.env.step(action)

            score += reward
            state = next_state

            if done:
                return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Deep Reinforcement Learning agent, or group of agents, which "
                                                 "is a double jointed arm robot with the goal of moving the end of the "
                                                 "into a constantly moving goal area. This is was developed as part of "
                                                 "solving the \"Continuous Control Project\" for the Udacity Deep "
                                                 "Reinforcement Learning Nanodegree.")
    parser.add_argument("--headless", action="store_true", help="Run the application in headless mode, i.e. "
                        "disable the visualization.")
    args = parser.parse_args()
    
    reacher = Reacher()
    score = reacher.play_episode()

    print("Score: {}".format(score))

    reacher.close()
