
import tensorflow as tf

class Network:
    def __init__(self, n_hid1=30, n_hid2=30, learning_rate=0.01):
        self.n_in = 784
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        self.n_out = 10

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.t = tf.placeholder(tf.float32, [None, 10])
        self.y = self._inference()

        self.loss = self._loss(self.y, self.t)
        self.train_step = self._training(self.loss, learning_rate=learning_rate)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _inference(self):
        def _weight_variable(shape):
            # randomは値固定とする
            initial = tf.truncated_normal(shape, mean=0.0, stddev=1.0)
            return tf.Variable(initial)

        def _bias_variable(shape):
            initial = tf.zeros(shape, dtype=tf.float32)
            return tf.Variable(initial)

        # InputLayer -> Hidden-1-Layer
        W1 = _weight_variable(shape=[self.n_in, self.n_hid1])
        b1 = _bias_variable(shape=[self.n_hid1])
        f1 = tf.matmul(self.x, W1) + b1
        # f1_out = self._LeakyReLU(f1)
        f1_out = tf.nn.sigmoid(f1)

        # Hidden-1-Layer -> Hidden-2-Layer
        W2 = _weight_variable(shape=[self.n_hid1, self.n_hid2])
        b2 = _bias_variable(shape=[self.n_hid2])
        f2 = tf.matmul(f1_out, W2) + b2
        # f2_out = self._LeakyReLU(f2)
        f2_out = tf.nn.sigmoid(f2)

        # Hidden-2-Layer -> OutputLayer
        W3 = _weight_variable(shape=[self.n_hid2, self.n_out])
        b3 = _bias_variable(shape=[self.n_out])
        f3 = tf.matmul(f2_out, W3) + b3
        f3_out = tf.nn.softmax(f3)

        y = f3_out
        return y

    def _loss(self, y, t):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y + 1.0**-8), reduction_indices=[1]))
        return cross_entropy

    def _training(self, loss, learning_rate, beta1=0.9, beta2=0.999):
        optimizer = \
             tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        train_step = optimizer.minimize(loss)
        return train_step

    def _LeakyReLU(self, f, a=0.2):
        return tf.maximum(f, f * a)



