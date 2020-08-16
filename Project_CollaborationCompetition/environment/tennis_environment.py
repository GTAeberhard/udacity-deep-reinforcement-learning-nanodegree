import os
from unityagents import UnityEnvironment


class TennisEnvironment:
    """
    Encapsulates the Tennis Unity environment and exposes a simple and familiar API
    which is common for interacting with Reinforcement Learning (RL) environments.

    The Tennis environment consists of two agents, which have the goal of hitting
    the ball over the net, while not hitting the ball out of bounds, or letting the
    ball hit the table. A reward of +0.1 is given if an agent hits the ball across
    net, and a reward of -0.01 if the ball hits the ground or goes out-of-bounds.

    The observation space consists of 8 continuous vairables and represents the
    position and velocity of the ball and the racket. Three observation frames are
    stacked together as input to the RL agent.

    The action space consists of 2 continuous continuous variable and represents the
    movements to the left and the right, as well as moving the paddles up and down,
    with values in the range of [-1, 1].

    The class supports both a graphical and headless version of the environment.

    Attributes
    ----------
        env: Variable which contains the UnityEnvironment class for this environment
        brain_name: Default name of the brain from the Unity Environment
        action_size (int): Size of the action space for a single agent
        state_size (int): Size of the state (observation) space for a single agent
        training_mode (bool): True if training mode is enabled, which runs the environment
            in an accelerated fashion for training
        num_agents (int): Number of agents in the environment
    """
    def __init__(self, training_mode=False, headless=False):
        """
        Initializes the Tennis Unity environment and sets up all of the class's
        attributes.

        Parameters
        ----------
            training_mode (bool): Set to True to enable training mode for the environment
            headless (bool): Set to True to enable the headless mode of the environment
        """
        headless_suffix = "_NoVis" if headless else ""
        root_path = os.path.dirname(os.path.realpath(__file__))
        self.env = UnityEnvironment(os.path.join(root_path, "Tennis_Linux{}/Tennis.x86_64".format(headless_suffix)))

        self.brain_name = self.env.brain_names[0]

        self.action_size = self.env.brains[self.brain_name].vector_action_space_size
        self.state_size = self.env.brains[self.brain_name].vector_observation_space_size * 3    # 3 frames stacked

        self.training_mode = training_mode

        env_info = self.env.reset(train_mode=self.training_mode)[self.brain_name]
        self.num_agents = len(env_info.agents)

    def close(self):
        """Properly closes the instantiated UnityEnvironment."""
        self.env.close()

    def reset(self):
        """Reset the environment for a new episode."""
        env_info = self.env.reset(train_mode=self.training_mode)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        """
        Take a single step in the environment and receive new information on the
        environment's current state. Note that the input and outputs are lists, where
        each element in the list correspond to the values for one of the agents.

        Parameters
        ----------
        actions: Actions which should be taken by the agents in the environment

        Returns
        -------
        (next_states, rewards, dones):
            next_states: The new state of the environment after having taken the action
            rewards: Rewards received for the agents after having take the action
            dones: Boolean which signifies if environment has ended, i.e. the episode is over
        """
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return (next_states, rewards, dones)
