import os
from unityagents import UnityEnvironment


class ReacherEnvironment:
    def __init__(self, training_mode=False, multiagent=False, headless=False):
        headless_suffix = "_NoVis" if headless else ""
        agent_type = "MultiAgent" if multiagent else "SingleAgent"
        root_path = os.path.dirname(os.path.realpath(__file__))
        self.env = UnityEnvironment(os.path.join(
            root_path, "Reacher{}_Linux{}/Reacher.x86_64".format(agent_type, headless_suffix))
        )

        self.brain_name = self.env.brain_names[0]

        self.action_size = self.env.brains[self.brain_name].vector_action_space_size
        self.state_size = self.env.brains[self.brain_name].vector_observation_space_size

        self.training_mode = training_mode

    def close(self):
        self.env.close()

    def reset(self):
        env_info = self.env.reset(train_mode=self.training_mode)[self.brain_name]
        self.num_agents = len(env_info.agents)
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        rewards = [0.1 if r > 0 else 0 for r in rewards]
        dones = env_info.local_done
        return (next_states, rewards, dones)
