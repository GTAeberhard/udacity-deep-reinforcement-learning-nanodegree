import unittest

from .agent import Agent


class DiscountedRewardsTest(unittest.TestCase):
    def test_simple_case(self):
        rewards = [[0, 1, 2, 3], [0, 1, 2, 3]]

        discounted_rewards = Agent._calculate_discounted_rewards(rewards, gamma=1)

        expected_discounted_rewards = [[6, 6, 5, 3], [6, 6, 5, 3]]
        self.assertListEqual(discounted_rewards, expected_discounted_rewards)

    def test_with_discount(self):
        rewards = [[0, 6, 4]]

        discounted_rewards = Agent._calculate_discounted_rewards(rewards, gamma=0.5)
        
        expected_discounted_rewards = [[4, 8, 4]]
        self.assertListEqual(discounted_rewards, expected_discounted_rewards)

    def test_with_discount_and_final_value(self):
        rewards = [[0, 6, 4]]

        discounted_rewards = Agent._calculate_discounted_rewards(rewards, final_values=[8], gamma=0.5)
        
        expected_discounted_rewards = [[5, 10, 8]]
        self.assertListEqual(discounted_rewards, expected_discounted_rewards)

if __name__ == "__main__":
    unittest.main()
