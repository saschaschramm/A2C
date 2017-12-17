import numpy as np
import tensorflow as tf
import random

def action_with_policy(policy):
    rand = random.uniform(0, 1)
    cumulated_sum = np.cumsum(policy)
    for i in range(0, len(cumulated_sum)):
        if rand <= cumulated_sum[i]:
            return i
    return 0

class Model():

    def __init__(self, policy, observation_space, action_space, nsteps, learning_rate, decay):
        self.learning_rate = learning_rate
        self.sess = tf.Session()

        self.actions = tf.placeholder(tf.int32, [nsteps])
        self.advantage = tf.placeholder(tf.float32, [nsteps])
        self.rewards = tf.placeholder(tf.float32, [nsteps])

        self.model_predict = policy(observation_space, action_space, 1, reuse=False)
        self.model_train = policy(observation_space, action_space, nsteps, reuse=True)

        action_masks = tf.one_hot(self.actions, action_space)

        self.loss_policy = tf.reduce_mean(
            -self.advantage * tf.reduce_sum(action_masks * tf.log(self.model_train.policy + 1e-13), 1))

        self.loss_value = tf.reduce_mean(tf.squared_difference(self.model_train.values, self.rewards))

        loss = self.loss_policy + self.loss_value
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=decay)

        self._train = optimizer.minimize(loss)
        tf.global_variables_initializer().run(session=self.sess)

    def predict(self, observation):
        policy, values = self.sess.run([self.model_predict.policy, self.model_predict.values],
                                       {self.model_predict.inputs: [observation]})
        action = action_with_policy(policy)
        return action, values[0]

    def predict_value(self, observation):
        return self.sess.run(self.model_predict.values, {self.model_predict.inputs: [observation]})

    def train(self, observations, rewards, actions, values):
        advantage = rewards - values
        policy_loss, value_loss, _ = self.sess.run(
            [self.loss_policy, self.loss_value, self._train],
            {
                self.model_train.inputs: observations,
                self.actions: actions,
                self.advantage: advantage,
                self.rewards: rewards
             }
        )
        return policy_loss, value_loss
