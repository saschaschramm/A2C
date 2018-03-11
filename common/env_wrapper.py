import numpy as np

class EnvWrapper:
    def __init__(self, env):
        self.env = env
        if env.observation_space.n == 64:
            self.observation_space = (8, 8)
        elif env.observation_space.n == 16:
            self.observation_space = (4, 4)
        else:
            raise NotImplementedError

        self.action_space = 4

    def grid_from_observation(self, observation):
        grid = np.zeros(self.observation_space)
        position = np.unravel_index(observation, self.observation_space)
        grid[position] = 1
        return grid

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)

        if done:
            observation = self.env.reset()

        grid = self.grid_from_observation(observation)
        return grid, reward, done

    def reset(self):
        observation = self.env.reset()
        grid = self.grid_from_observation(observation)
        return grid