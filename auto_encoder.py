from glob import *
from PIL import Image
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os

w_enc = None
b_enc = None

def inference(in_data, n_hidden):
    n_in = in_data[0].size

    x = tf.placeholder("float", [None, n_in])
    global w_enc
    w_enc = tf.Variable(tf.random_normal([n_in, n_hidden], mean=0.0, stddev=0.01))
    w_dec = tf.Variable(tf.random_normal([n_hidden, n_in], mean=0.0, stddev=0.01))
    # w_dec = tf.transpose(w_enc) # if you use tied weights
    global b_enc
    b_enc = tf.Variable(tf.zeros([n_hidden]))
    b_dec = tf.Variable(tf.zeros([n_in]))

    encoded = tf.sigmoid(tf.matmul(x, w_enc) + b_enc)
    decoded = tf.sigmoid(tf.matmul(encoded, w_dec) + b_dec)

    return x, decoded

input_path = "../dataset/*.jpg"
files = glob(input_path)

def loss(y, t):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
    return cross_entropy

def training(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_step = optimizer.minimize(loss)
    return train_step

images = []
for file in files:
    print("import! file:", file)
    image = np.array(Image.open(file).resize((1000,1000))).reshape(-1)
    images.append(image)

images = np.array(images)
# Variables
n_in = images[0].size
n_hidden = 2

x, y = inference(images, n_hidden)
t = tf.placeholder("float", [None, n_in])
image_tf = tf.placeholder("float", [None, len(images[0])])
feature = tf.Variable("float", [n_hidden])

loss = loss(y, t)
train_step = training(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

epochs = 10000
batch_size = 1
n_batches = int(len(images) / batch_size)

for epoch in range(epochs):
    X_ = shuffle(images)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: X_[start:end]
        })
        loss_ = loss.eval(session=sess, feed_dict={
            x: X_[start:end],
            t: X_[start:end]
        })

    print("epoch:", epoch, " loss:", loss_)
print(sess.run(w_enc))
print(sess.run(b_enc))
feature = tf.matmul(image_tf, w_enc)

# 特徴量抽出
plt.figure()
for i in range(len(images)):
    # imageを[1, n]の２次元配列に
    Z = images[i].reshape(1,-1)

    # [1, n] * [n, 2]の行列かけざん！
    y_ = sess.run(feature, feed_dict={image_tf: Z})
    print("image=", str(files[i]), "feature:", y_)

    if os.path.basename(files[i])[:3] == "yui":
        plt.plot(y_[0,0], y_[0,1], "yo")
    if os.path.basename(files[i])[:5] == "azusa":
        plt.plot(y_[0,0], y_[0,1], "bo")
    if os.path.basename(files[i])[:5] == "apple":
        plt.plot(y_[0,0], y_[0,1], "ro")

plt.show()
