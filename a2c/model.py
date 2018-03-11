import tensorflow as tf

def sample(probs):
    random_uniform = tf.random_uniform(tf.shape(probs))
    scaled_random_uniform = tf.log(random_uniform) / probs
    return tf.argmax(scaled_random_uniform, axis=1)

class Model():

    def __init__(self, policy, observation_space, action_space, nsteps, learning_rate, decay):
        self.learning_rate = learning_rate
        self.sess = tf.Session()

        self.actions = tf.placeholder(tf.int32, [nsteps])
        self.advantage = tf.placeholder(tf.float32, [nsteps])
        self.rewards = tf.placeholder(tf.float32, [nsteps])

        self.model = policy(observation_space, action_space)

        logits = tf.reduce_sum(tf.one_hot(self.actions, action_space) *
                               tf.log(self.model.policy + 1e-13), axis=1)

        self.loss_policy = -tf.reduce_mean(self.advantage * logits)

        self.loss_value = tf.reduce_mean(tf.squared_difference(self.model.values, self.rewards))

        loss = self.loss_policy + self.loss_value
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=decay).minimize(loss)

        self.sampled_action = sample(self.model.policy)
        tf.global_variables_initializer().run(session=self.sess)

    def predict(self, observation):
        actions, values = self.sess.run([self.sampled_action, self.model.values],
                                       {self.model.inputs: [observation]})
        return actions[0], values[0]

    def predict_value(self, observation):
        return self.sess.run(self.model.values, {self.model.inputs: [observation]})

    def train(self, observations, rewards, actions, values):
        advantage = rewards - values
        loss_policy, loss_value, _ = self.sess.run(
            [self.loss_policy, self.loss_value, self.optimizer],
            {
                self.model.inputs: observations,
                self.actions: actions,
                self.advantage: advantage,
                self.rewards: rewards
             }
        )
        return loss_policy, loss_value
