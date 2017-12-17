import tensorflow as tf
import numpy as np
import random

step_count = 0
scores = []

def print_score(reward, limit=100000):
    global scores
    global step_count

    scores.append(reward)

    if len(scores) == limit:
        print("{}: {}".format(step_count, sum(scores)))
        step_count += 1
        scores = []

def global_seed():
    tf.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)


def discount(rewards, dones, discount_rate):
    discounted = []
    total_return = 0

    for reward, done in zip(rewards[::-1], dones[::-1]):
        if done:
            total_return = reward
        else:
            total_return = reward + discount_rate * total_return
        discounted.append(total_return)
    return np.asarray(discounted[::-1])