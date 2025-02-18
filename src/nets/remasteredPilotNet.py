import tensorflow.compat.v1 as tf

def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

class RemasteredPilotNet(object):
    """An End-to-End remastered PilotNet to output encountered entities given input images of the road ahead"""

    def __init__(self):
        self.image_input = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
        # use for training loss computation
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        # model parameters
        # self.model_params = []

        # first convolutional layer
        W_conv1 = _weight_variable([5, 5, 3, 24])
        b_conv1 = _bias_variable([24])

        h_conv1 = tf.nn.relu(_conv2d(self.image_input, W_conv1, 2) + b_conv1)

        # second convolutional layer
        W_conv2 = _weight_variable([5, 5, 24, 36])
        b_conv2 = _bias_variable([36])

        h_conv2 = tf.nn.relu(_conv2d(h_conv1, W_conv2, 2) + b_conv2)

        # third convolutional layer
        W_conv3 = _weight_variable([5, 5, 36, 48])
        b_conv3 = _bias_variable([48])

        h_conv3 = tf.nn.relu(_conv2d(h_conv2, W_conv3, 2) + b_conv3)

        # fourth convolutional layer
        W_conv4 = _weight_variable([3, 3, 48, 64])
        b_conv4 = _bias_variable([64])

        h_conv4 = tf.nn.relu(_conv2d(h_conv3, W_conv4, 1) + b_conv4)

        # fifth convolutional layer
        W_conv5 = _weight_variable([3, 3, 64, 64])
        b_conv5 = _bias_variable([64])

        h_conv5 = tf.nn.relu(_conv2d(h_conv4, W_conv5, 1) + b_conv5)

        # FCL 1
        W_fc1 = _weight_variable([1152, 1164])
        b_fc1 = _bias_variable([1164])

        h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # FCL 2
        W_fc2 = _weight_variable([1164, 100])
        b_fc2 = _bias_variable([100])

        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)

        # FCL 3
        W_fc3 = _weight_variable([100, 50])
        b_fc3 = _bias_variable([50])

        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

        h_fc3_drop = tf.nn.dropout(h_fc3, self.keep_prob)

        # FCL 3
        W_fc4 = _weight_variable([50, 10])
        b_fc4 = _bias_variable([10])

        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

        h_fc4_drop = tf.nn.dropout(h_fc4, self.keep_prob)

        # Output
        W_fc5 = _weight_variable([10, 1])
        b_fc5 = _bias_variable([1])

        self.steering = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2)  # scale the atan output


