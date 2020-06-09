import unittest

from reacher import Reacher


class ReacherInferenceTest(unittest.TestCase):
    def test_achieve_basic_score(self):
        reacher = Reacher(headless=True)
        score = reacher.play_episode()
        reacher.close()

        self.assertGreaterEqual(score, 0)


if __name__ == "__main__":
    unittest.main()
