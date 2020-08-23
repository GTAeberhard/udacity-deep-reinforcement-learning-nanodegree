import os
import unittest
import numpy as np

from tennis import Tennis


class TennisInferenceTest(unittest.TestCase):
    def test_achieve_basic_score(self):
        tennis = Tennis(headless=True, device="cpu")
        actor_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights_actor.pth")
        critic_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights_critic.pth")
        tennis.load_weights(actor_weights, critic_weights)
        score = tennis.play_episode()
        tennis.close()

        self.assertGreaterEqual(np.max(score), 0.5)


if __name__ == "__main__":
    unittest.main()
