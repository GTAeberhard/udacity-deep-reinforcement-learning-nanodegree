import unittest

from banana_navigation import BananaNavigation


class BananaNavigationInferenceTest(unittest.TestCase):
    def test_achieve_basic_score(self):
        banana_navigation = BananaNavigation(headless=True, device="cpu")
        banana_navigation.load_weights()
        score = banana_navigation.play_episode()

        self.assertGreaterEqual(score, 10)


if __name__ == "__main__":
    unittest.main()
