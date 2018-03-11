import tensorflow as tf
import numpy as np
import random
import gym
from gym.envs.registration import register

from env_wrapper import EnvWrapper

def init_env(map_name):
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': map_name,
                'is_slippery': False}
    )

    env = gym.make('FrozenLakeNotSlippery-v0')
    return EnvWrapper(env)


def global_seed():
    tf.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)