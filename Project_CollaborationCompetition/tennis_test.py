import os
import unittest

from tennis import Tennis


class TennisInferenceTest(unittest.TestCase):
    @unittest.skip
    def test_achieve_basic_score(self):
        tennis = Tennis(headless=True, device="cpu")
        # default_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights.pth")
        # tennis.load_weights(default_weights)
        score = tennis.play_episode()
        tennis.close()

        self.assertGreaterEqual(score, 0)


if __name__ == "__main__":
    unittest.main()
