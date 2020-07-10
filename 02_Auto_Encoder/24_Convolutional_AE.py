import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
DATA_DIR = "/tmp/ML/MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Parameters
learning_rate = 0.001
N_EPISODES = 5
display_step = 1
examples_to_show = 10
bs = 256

# tf input graph
X = tf.placeholder(tf.float32, [None, 784])
batch_size = tf.shape(X)[0]

# Store weights and biases

weights = {
    # convolution 1, 3x3 filter
    'wec1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
    # convolution 2, 3x3 filter
    'wec2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # fully connected layer 1, encoder
    'wef1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # fully connected layer 2, encoder
    'wef2': tf.Variable(tf.random_normal([1024, 256])),
    # fully connected layer 1, decoder
    'wdf1': tf.Variable(tf.random_normal([256, 1024])),
    # fully connected layer 2, decoder
    'wdf2': tf.Variable(tf.random_normal([1024, 7*7*64])),
    # deconvolution 1, 3x3 filter
    'wdd1': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    # deconvolution 2, 3x3 filter
    'wdd2': tf.Variable(tf.random_normal([3, 3, 1, 32]))
}

biases = {
    'bec1': tf.Variable(tf.random_normal([32])),
    'bec2': tf.Variable(tf.random_normal([64])),
    'bef1': tf.Variable(tf.random_normal([1024])),
    'bef2': tf.Variable(tf.random_normal([256])),
    'bdf1': tf.Variable(tf.random_normal([1024])),
    'bdf2': tf.Variable(tf.random_normal([7*7*64])),
    'bdd1': tf.Variable(tf.random_normal([32])),
    'bdd2': tf.Variable(tf.random_normal([1]))
}

def activation(_input, activation='sigmoid'):
    if activation == 'relu':
        return tf.nn.relu(_input)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(_input)


# create model

reshaped_x = tf.reshape(X, (-1, 28, 28, 1))
print(reshaped_x.shape)

conv1 = tf.nn.conv2d(reshaped_x, weights['wec1'], strides=[1, 2, 2, 1], padding='SAME')
conv1 = tf.contrib.layers.batch_norm(conv1, epsilon=1e-5) # batch norm added
conv1 = activation(tf.add(conv1, biases['bec1']))
print(conv1.shape)

conv2 = tf.nn.conv2d(conv1, weights['wec2'], strides=[1, 2, 2, 1], padding='SAME')
conv2 = tf.contrib.layers.batch_norm(conv2, epsilon=1e-5) # batch norm added
conv2 = activation(tf.add(conv2, biases['bec2']))
print(conv2.shape)

fce1 = tf.reshape(conv2, (-1, 7*7*64))
fce1 = activation(tf.add(tf.matmul(fce1, weights['wef1']), biases['bef1']))
print(fce1.shape)

encode = activation(tf.add(tf.matmul(fce1, weights['wef2']), biases['bef2']))
print(encode.shape)

fcd1 = activation(tf.add(tf.matmul(encode, weights['wdf1']), biases['bdf1']))
print(fcd1.shape)

fcd2 = activation(tf.add(tf.matmul(fcd1, weights['wdf2']), biases['bdf2']))
fcd2 = tf.reshape(fcd2, (-1, 7, 7, 64))
print(fcd2.shape)

deconv1 = tf.nn.conv2d_transpose(fcd2, 
                                 weights['wdd1'], 
                                 output_shape=tf.stack([batch_size, 14, 14, 32]), 
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')
deconv1 = tf.reshape(deconv1, (batch_size, 14, 14, 32))
deconv1 = tf.contrib.layers.batch_norm(deconv1, epsilon=1e-5) # batch norm added
deconv1 = activation(tf.add(deconv1, biases['bdd1']))
print(deconv1.shape)

deconv2 = tf.nn.conv2d_transpose(deconv1,
                                 weights['wdd2'],
                                 output_shape=tf.stack([batch_size, 28, 28, 1]),
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')
deconv2 = tf.reshape(deconv2, (batch_size, 28, 28, 1))
deconv2 = tf.contrib.layers.batch_norm(deconv2, epsilon=1e-5) # batch norm added
deconv2 = activation(tf.add(deconv2, biases['bdd2']))
print(deconv2.shape)

decode = tf.reshape(deconv2, (-1, 784))

cost = tf.reduce_mean(tf.square(decode - X))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/bs)
    
    for epoch in range(N_EPISODES):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(bs)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
            print("i:{0}, Cost:{1}".format(i, c))
        if epoch % display_step == 0:
            print('Epoch:', (epoch+1), 'Cost:', c)
    print('Optimzation finished!')
    
    save_path = saver.save(sess, "Saved_models/cnn_auto_mnist_sigmoid/conv_ae_model.ckpt")
    print("Model saved in file: %s" % save_path)
    
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "Saved_models/cnn_auto_mnist_sigmoid/conv_ae_model.ckpt")
    print("Model restored.")
    
    encode_decode = sess.run(decode, feed_dict={X: mnist.test.images[:examples_to_show]}) # first 10 rows from the test dataset

# compare the original images with the reconstructions
f, a = plt.subplots(2, 10, figsize=(10,2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
plt.draw()    
    
plt.show()

img_org = plt.imshow(np.reshape(x, (28,28)))
plt.show()

img_pred = plt.imshow(np.reshape(out, (28,28)))
plt.show()

