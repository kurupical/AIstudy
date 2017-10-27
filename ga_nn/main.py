
from tensorflow.examples.tutorials.mnist import input_data

from network import *

# データ
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# ネットワークの定義
net = Network()

# 学習
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    net.sess.run(net.train_step, feed_dict={
        net.x: batch_xs,
        net.t: batch_ys
    })
    if (i+1) % 100 == 0:
        y_pred = net.y.eval(session=net.sess, feed_dict={
            net.x: mnist.test.images,
            net.t: mnist.test.labels
        })

        val_loss = net.loss.eval(session=net.sess, feed_dict={
            net.x: mnist.test.images,
            net.y: y_pred,
            net.t: mnist.test.labels
        })

        accuracy = net.sess.run(net.accuracy, feed_dict={
            net.y: y_pred,
            net.t: mnist.test.labels
        })
        print("i={}, val_loss={:.6f}, accuracy={:4f}".format(i+1, val_loss, accuracy))