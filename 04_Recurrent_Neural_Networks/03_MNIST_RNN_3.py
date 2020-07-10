from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

DATA_DIR = "/tmp/ML/MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

SAVE_DIR = "/tmp/ML/14_Recurrent_Neural_Networks"

# Training Parameters
learning_rate = 0.001
training_steps = 2000
batch_size = 100
display_step = 200

"""
  To classify images using a recurrent neural network, we consider every image
  row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
  handle 28 sequences of 28 steps for every sample.
  RNN 은 순서가 있는 자료를 다루므로,
  한 번에 입력받는 갯수와, 총 몇 단계로 이루어져있는 데이터를 받을지를 설정해야합니다.
  이를 위해 가로 픽셀수를 INPUT_SIZE 으로, 세로 픽셀수를 입력 단계인 TIME_STEP 으로 설정하였습니다.
"""

# Network Parameters
INPUT_SIZE = 28              # MNIST data input (img shape: 28*28)
TIME_STEP  = 28              # TIME_STEP
H_SIZE_01  = 128             # hidden layer num of features
n_class    = 10              # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, TIME_STEP, INPUT_SIZE])
Y = tf.placeholder("float", [None, n_class])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([H_SIZE_01, n_class]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_class]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, TIME_STEP, n_input)
    # Required shape: 'TIME_STEP' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'TIME_STEP' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, TIME_STEP, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(H_SIZE_01, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_xs = batch_xs.reshape((batch_size, TIME_STEP, INPUT_SIZE))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_xs,
                                                                 Y: batch_ys})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_batch_size = 128
    test_xs = mnist.test.images[:test_batch_size].reshape((-1, TIME_STEP, INPUT_SIZE))
    test_ys = mnist.test.labels[:test_batch_size]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))
    


    #########
    # 결과 확인 (matplot)
    ######
    test_batch_size = 1024
    test_xs = mnist.test.images[:test_batch_size].reshape((-1, TIME_STEP, INPUT_SIZE))
    test_ys = mnist.test.labels[:test_batch_size]
    labels = sess.run(prediction,feed_dict={X: test_xs, Y: test_ys})

    fig = plt.figure()
    for i in range(60):
        subplot = fig.add_subplot(4, 15, i + 1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title('%d' % np.argmax(labels[i]))
        subplot.imshow(mnist.test.images[i].reshape((28, 28)),
                       cmap=plt.cm.gray_r)

    plt.show()


    # 세션을 닫습니다.
    sess.close()


    # Step 10. Tune hyperparameters:
    # Step 11. Deploy/predict new outcomes:


    