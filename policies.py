import tensorflow as tf

class PolicyFullyConnected(object):

    def __init__(self, observation_space, action_space, nsteps, reuse=False):
        height, width = observation_space
        self.inputs = tf.placeholder(tf.float32, (nsteps, height, width))
        with tf.variable_scope("model", reuse=reuse):
            inputs_reshaped = tf.reshape(self.inputs, [nsteps, width * height])
            hidden = tf.contrib.layers.fully_connected(inputs=inputs_reshaped, num_outputs=100, activation_fn=None)
            logits = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=action_space, activation_fn=None)
            self.values = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=1, activation_fn=None)[:, 0]
        self.policy = tf.nn.softmax(logits)
