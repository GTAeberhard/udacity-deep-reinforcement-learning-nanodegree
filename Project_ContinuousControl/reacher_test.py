import os
import unittest

from reacher import Reacher


class ReacherInferenceTest(unittest.TestCase):
    def setUp(self):
        self.ddpg_actor_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights_actor_ddpg.pth")
        self.ddpg_critic_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights_critic_ddpg.pth")

    def test_achieve_basic_score_single_agent(self):
        reacher = Reacher(agent_type="ddpg", headless=True, device="cpu")
        reacher.load_weights(self.ddpg_actor_weights, self.ddpg_critic_weights)
        score = reacher.play_episode()
        reacher.close()

        self.assertGreaterEqual(score, 30)

    def test_achieve_basic_score_multi_agent(self):
        reacher = Reacher(agent_type="ddpg", headless=True, device="cpu", multiagent=True)
        reacher.load_weights(self.ddpg_actor_weights, self.ddpg_critic_weights)
        score = reacher.play_episode()
        reacher.close()

        self.assertGreaterEqual(score, 30)

if __name__ == "__main__":
    unittest.main()
