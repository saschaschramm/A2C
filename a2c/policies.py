import tensorflow as tf

class PolicyFullyConnected(object):

    def __init__(self, observation_space, action_space):
        height, width = observation_space
        self.inputs = tf.placeholder(tf.float32, (None, height, width))
        inputs_reshaped = tf.reshape(self.inputs, [tf.shape(self.inputs)[0], height*width])
        hidden = tf.layers.dense(inputs=inputs_reshaped, units=256, activation=None)
        logits = tf.layers.dense(inputs=hidden, units=action_space, activation=None)
        self.values = tf.layers.dense(inputs=hidden, units=1, activation=None)[:, 0]
        self.policy = tf.nn.softmax(logits)