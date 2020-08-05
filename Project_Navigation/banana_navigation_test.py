import os
import unittest

from .banana_navigation import BananaNavigation


class BananaNavigationInferenceTest(unittest.TestCase):
    def test_achieve_basic_score(self):
        banana_navigation = BananaNavigation(headless=True, device="cpu")
        default_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights.pth")
        banana_navigation.load_weights(default_weights)
        score = banana_navigation.play_episode()
        banana_navigation.close()

        self.assertGreaterEqual(score, 10)


if __name__ == "__main__":
    unittest.main()
