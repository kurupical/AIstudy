# common lib
import tensorflow as tf

# original lib

class DQN:
    '''
    DQNのネットワークを定義
    '''
    def __init__(self, n_l1hidden, n_l2hidden, n_l3hidden, n_in, n_out, learning_rate=0.001):
        '''
        n_in: 入力(ゲームの状態)のノード数
        n_out: 出力(Q(s,a))
        '''

        self.n_l1hidden = n_l1hidden
        self.n_l2hidden = n_l2hidden
        self.n_l3hidden = n_l3hidden
        self.n_in = n_in
        self.n_out = n_out
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_in])
        self.t = tf.placeholder(tf.float32, shape=[None, self.n_out])

        self.y = self._inference()
        self.loss = self._loss(self.y, self.t)
        self.train_step = self._training(self.loss, learning_rate=learning_rate)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # tf.summary.FileWriter(config_path, self.sess.graph)

    def _inference(self):
        '''
        入力層 -> 隠れ層２ -> 出力層
        '''
        def _weight_variable(shape):
            initial = tf.truncated_normal(shape, mean=0.0, stddev=1.0)
            # initial = tf.random_uniform(shape, minval=-1, maxval=1)
            return tf.Variable(initial)

        def _bias_variable(shape):
            initial = tf.zeros(shape, dtype=tf.float32)
            return tf.Variable(initial)

        # inputlayer -> hiddenlayer-1
        W1 = _weight_variable(shape=[self.n_in, self.n_l1hidden])
        b1 = _bias_variable(shape=[self.n_l1hidden])
        f1 = tf.matmul(self.x, W1) + b1
        f1_out = self._LeakyReLU(f1)

        # hiddenlayer-1 -> hiddenlayer-2
        W2 = _weight_variable(shape=[self.n_l1hidden, self.n_l2hidden])
        b2 = _bias_variable(shape=[self.n_l2hidden])
        f2 = tf.matmul(f1_out, W2) + b2
        f2_out = self._LeakyReLU(f2)

        # hiddenlayer-2 -> hiddenlayer-3
        W3 = _weight_variable(shape=[self.n_l2hidden, self.n_l1hidden])
        b3 = _bias_variable(shape=[self.n_l3hidden])
        f3 = tf.matmul(f2_out, W3) + b3
        f3_out = self._LeakyReLU(f3)

        # hidden-layer3 -> outputlayer
        W4 = _weight_variable(shape=[self.n_l3hidden, self.n_out])
        b4 = _bias_variable(shape=[self.n_out])
        f4 = tf.matmul(f3_out, W4) + b4

        # output
        y = f4

        return y

    def _loss(self, y, t):
        mse = tf.reduce_mean(tf.square(y - t))
        return mse

    def _training(self, loss, learning_rate, beta1=0.9, beta2=0.999):
        #optimizer = \
        #     tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        train_step = optimizer.minimize(loss)
        return train_step

    def _LeakyReLU(self, f, a=0.2):
        return tf.maximum(f, f*a)

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)
