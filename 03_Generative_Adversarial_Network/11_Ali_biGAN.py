import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

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

SAVE_DIR = "/tmp/ML/13_Generative_Adversarial_Network/11_Ali_Bi_GAN"

# Define Hyper Parameters
N_EPISODES = 10000
mb_size = 32

INPUT_SIZE = mnist.train.images.shape[1]
# OUTPUT_SIZE = mnist.train.labels.shape[1]
# INPUT_SIZE = 784
NOISE_SIZE = 128
H_SIZE_01 = 256
lr = 1e-3
d_steps = 3
row_plot = 8
column_plot = 5
N_SAMPLE = row_plot * column_plot

def plot(samples):
    fig = plt.figure(figsize=(row_plot, column_plot))
    gs = gridspec.GridSpec(row_plot, column_plot)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def log(x):
    return tf.log(x + 1e-8)

X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE])

"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Generator Variables
W01_Q = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Q = tf.Variable(xavier_init([H_SIZE_01, NOISE_SIZE]))
B01_Q = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Q = tf.Variable(tf.zeros(shape=[NOISE_SIZE]))

W01_P = tf.Variable(xavier_init([NOISE_SIZE, H_SIZE_01]))
W02_P = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_P = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_P = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))

# Discriminator Variables
W01_Dis = tf.Variable(xavier_init([INPUT_SIZE + NOISE_SIZE, H_SIZE_01]))
W02_Dis = tf.Variable(xavier_init([H_SIZE_01, 1]))
B01_Dis = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Dis = tf.Variable(tf.zeros(shape=[1]))
"""

W01_Q   = tf.get_variable("W01_Q", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Q   = tf.get_variable("W02_Q", shape=[H_SIZE_01, NOISE_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Q   = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Q   = tf.Variable(tf.random_normal([NOISE_SIZE]))

W01_P   = tf.get_variable("W01_P", shape=[NOISE_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_P   = tf.get_variable("W02_P", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_P   = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_P   = tf.Variable(tf.random_normal([INPUT_SIZE]))

W01_Dis = tf.get_variable("W01_Dis", shape=[INPUT_SIZE + NOISE_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis = tf.get_variable("W02_Dis", shape=[H_SIZE_01, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Dis = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis = tf.Variable(tf.random_normal([1]))

def Q(X):
    _LAY01_Q = tf.nn.relu(tf.matmul(X, W01_Q) + B01_Q)
    output_Q = tf.matmul(_LAY01_Q, W02_Q) + B02_Q
    return output_Q

def P(z):
    _LAY01_P = tf.nn.relu(tf.matmul(z, W01_P) + B01_P)
    output_P = tf.nn.sigmoid(tf.matmul(_LAY01_P, W02_P) + B02_P)
    return output_P

def DISCRIMINATOR(X, z):
    inputs = tf.concat([X, z], axis=1)
    _LAY01_Dis = tf.nn.relu(tf.matmul(inputs, W01_Dis) + B01_Dis)
    output_Dis = tf.nn.sigmoid(tf.matmul(_LAY01_Dis, W02_Dis) + B02_Dis)
    return output_Dis

def GET_NOISE(BATCH_SIZE, NOISE_SIZE):
#    return np.random.uniform(-1., 1., size=[BATCH_SIZE, NOISE_SIZE])
    return np.random.normal(-1., 1., size=[BATCH_SIZE, NOISE_SIZE])

G_var_list = [W01_Q, W02_Q, B01_Q, B02_Q, W01_P, W02_P, B01_P, B02_P]
D_var_list = [W01_Dis, W02_Dis, B01_Dis, B02_Dis]

z_hat = Q(X)
X_hat = P(z)

D_enc = DISCRIMINATOR(X, z_hat)
D_gen = DISCRIMINATOR(X_hat, z)

D_loss = -tf.reduce_mean(log(D_enc) + log(1 - D_gen))
G_loss = -tf.reduce_mean(log(D_gen) + log(1 - D_enc))

D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(D_loss, var_list=D_var_list))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=G_var_list))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
i = 0

for episode in range(N_EPISODES):
    X_mb, _ = mnist.train.next_batch(mb_size)
    z_mb = GET_NOISE(mb_size, NOISE_SIZE)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss], feed_dict={X: X_mb, z: z_mb}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={X: X_mb, z: z_mb}
    )

    if episode % 1000 == 0:
        print("[Episode : {:>5}] [D_loss: {:2.5f}] [G_loss: {:2.5f}]"
              .format(episode, D_loss_curr, G_loss_curr))

        samples = sess.run(X_hat, feed_dict={z: GET_NOISE(N_SAMPLE, NOISE_SIZE)})

        fig = plot(samples)
        plt.savefig(SAVE_DIR + '/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
