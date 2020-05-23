from unityagents import UnityEnvironment


class BananaEnvironment:
    def __init__(self, training_mode=False, headless=False):
        headless_suffix = "_NoVis" if headless else ""
        self.env = UnityEnvironment("environment/Banana_Linux{}/Banana.x86_64".format(headless_suffix))

        # Get the default brain
        self.brain_name = self.env.brain_names[0]

        self.action_size = self.env.brains[self.brain_name].vector_action_space_size
        self.state_size = self.env.brains[self.brain_name].vector_observation_space_size

        self.training_mode = training_mode

    def reset(self):
        env_info = self.env.reset(train_mode=self.training_mode)[self.brain_name]
        return env_info.vector_observations[0]

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return (next_state, reward, done)
