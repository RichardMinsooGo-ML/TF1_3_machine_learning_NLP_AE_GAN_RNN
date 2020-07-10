import tensorflow as tf
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

# Define Hyper Parameters
learning_rate = 0.001
N_EPISODES = 10
batch_size = 100

"""
  To classify images using a recurrent neural network, we consider every image
  row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
  handle 28 sequences of 28 steps for every sample.
  RNN 은 순서가 있는 자료를 다루므로,
  한 번에 입력받는 갯수와, 총 몇 단계로 이루어져있는 데이터를 받을지를 설정해야합니다.
  이를 위해 가로 픽셀수를 INPUT_SIZE 으로, 세로 픽셀수를 입력 단계인 TIME_STEP 으로 설정하였습니다.
"""
INPUT_SIZE = 28
TIME_STEP = 28
H_SIZE_01 = 128
n_class = 10

# Define Placeholder
X = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])
Y = tf.placeholder(tf.float32, [None, n_class])

W01_m = tf.Variable(tf.random_normal([H_SIZE_01, n_class]))
B01_m = tf.Variable(tf.random_normal([n_class]))

# RNN 에 학습에 사용할 셀을 생성합니다
# 다음 함수들을 사용하면 다른 구조의 셀로 간단하게 변경할 수 있습니다
# BasicRNNCell,BasicLSTMCell,GRUCell
cell = tf.nn.rnn_cell.BasicRNNCell(H_SIZE_01)

# RNN 신경망을 생성합니다
# 원래는 다음과 같은 과정을 거쳐야 하지만
# states = tf.zeros(batch_size)
# for i in range(TIME_STEP):
#     outputs, states = cell(X[[:, i]], states)
# ...
# 다음처럼 tf.nn.dynamic_rnn 함수를 사용하면
# CNN 의 tf.nn.conv2d 함수처럼 간단하게 RNN 신경망을 만들어줍니다.
# 겁나 매직!!
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# 결과를 Y의 다음 형식과 바꿔야 하기 때문에
# Y : [batch_size, n_class]
# outputs 의 형태를 이에 맞춰 변경해야합니다.
# outputs : [batch_size, TIME_STEP, H_SIZE_01]
#        -> [TIME_STEP, batch_size, H_SIZE_01]
#        -> [batch_size, H_SIZE_01]
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
Pred_m = tf.matmul(outputs, W01_m) + B01_m

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_m, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)

for episode in range(N_EPISODES):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # X 데이터를 RNN 입력 데이터에 맞게 [batch_size, TIME_STEP, INPUT_SIZE] 형태로 변환합니다.
        batch_xs = batch_xs.reshape((batch_size, TIME_STEP, INPUT_SIZE))

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('episode:', '%04d' % (episode + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('Optimization Completed!')

#########
# 결과 확인
######
is_correct = tf.equal(tf.argmax(Pred_m, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, TIME_STEP, INPUT_SIZE)
test_ys = mnist.test.labels

print('Accuracy:', sess.run(accuracy,
                       feed_dict={X: test_xs, Y: test_ys}))


#########
# 결과 확인 (matplot)
######
labels = sess.run(Pred_m,
                  feed_dict={X: mnist.test.images.reshape(-1, 28, 28),
                             Y: mnist.test.labels})

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

