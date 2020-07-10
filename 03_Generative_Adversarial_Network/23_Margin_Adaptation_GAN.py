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

SAVE_DIR = "/tmp/ML/13_Generative_Adversarial_Network/23_Ma_GAN"

# Define Hyper Parameters
N_EPISODES = 10000
mb_size = 32

# Parameters for the Networks
INPUT_SIZE = 784
NOISE_SIZE = 128
H_SIZE_01 = 256
lr = 5e-4
n_iter = 1000
n_epoch = 10
N = n_iter * mb_size  # N data per epoch

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE])
m = tf.placeholder(tf.float32)

"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Generator Variables
W01_Gen = tf.Variable(xavier_init([NOISE_SIZE, H_SIZE_01]))
W02_Gen = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_Gen = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Gen = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))

# Discriminator Variables
W01_Dis = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Dis = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_Dis = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Dis = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))
"""

W01_Gen = tf.get_variable("W01_Gen", shape=[NOISE_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Gen = tf.get_variable("W02_Gen", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Gen = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Gen = tf.Variable(tf.random_normal([INPUT_SIZE]))

W01_Dis = tf.get_variable("W01_Dis", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis = tf.get_variable("W02_Dis", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Dis = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis = tf.Variable(tf.random_normal([INPUT_SIZE]))

# Build Generator Network.
def GENERATOR(z):
    _LAY01_Gen = tf.nn.relu(tf.matmul(z, W01_Gen) + B01_Gen)
    output_Gen = tf.nn.sigmoid(tf.matmul(_LAY01_Gen, W02_Gen) + B02_Gen)
    return output_Gen

def DISCRIMINATOR(X):
    _LAY01_Dis = tf.nn.relu(tf.matmul(X, W01_Dis) + B01_Dis)
    X_recon = tf.matmul(_LAY01_Dis, W02_Dis) + B02_Dis
    output_Dis = tf.reduce_sum((X - X_recon)**2, 1)
    return output_Dis

def GET_NOISE(BATCH_SIZE, NOISE_SIZE):
#    return np.random.uniform(-1., 1., size=[BATCH_SIZE, NOISE_SIZE])
    return np.random.normal(-1., 1., size=[BATCH_SIZE, NOISE_SIZE])

G_var_list = [W01_Gen, W02_Gen, B01_Gen, B02_Gen]
D_var_list = [W01_Dis, W02_Dis, B01_Dis, B02_Dis]

G_sample = GENERATOR(z)

D_real = DISCRIMINATOR(X)
D_fake = DISCRIMINATOR(G_sample)

D_recon_loss = tf.reduce_mean(D_real)
D_loss = tf.reduce_mean(D_real + tf.maximum(0., m - D_fake))
G_loss = tf.reduce_mean(D_fake)

D_recon_solver = (tf.train.AdamOptimizer(learning_rate=lr)
                  .minimize(D_recon_loss, var_list=D_var_list))
D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(D_loss, var_list=D_var_list))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=G_var_list))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Pretrain
for it in range(2*n_iter):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_recon_loss_curr = sess.run(
        [D_recon_solver, D_recon_loss], feed_dict={X: X_mb}
    )

    if it % 1000 == 0:
        print('Iter-{}; Pretrained D loss: {:.4}'.format(it, D_recon_loss_curr))


i = 0
# Initial margin, expected energy of real data
margin = sess.run(D_recon_loss, feed_dict={X: mnist.train.images})
s_z_before = np.inf

# GAN training
for t in range(n_epoch):
    s_x, s_z = 0., 0.

    for it in range(n_iter):
        X_mb, _ = mnist.train.next_batch(mb_size)
        z_mb = GET_NOISE(mb_size, NOISE_SIZE)

        _, D_loss_curr, D_real_curr = sess.run(
            [D_solver, D_loss, D_real], feed_dict={X: X_mb, z: z_mb, m: margin}
        )

        # Update real samples statistics
        s_x += np.sum(D_real_curr)

        _, G_loss_curr, D_fake_curr = sess.run(
            [G_solver, G_loss, D_fake],
            feed_dict={X: X_mb, z: GET_NOISE(mb_size, NOISE_SIZE), m: margin}
        )

        # Update fake samples statistics
        s_z += np.sum(D_fake_curr)

    # Update margin
    if (s_x / N < margin) and (s_x < s_z) and (s_z_before < s_z):
        margin = s_x / N

    s_z_before = s_z

    # Convergence measure
    Ex = s_x / N
    Ez = s_z / N
    L = Ex + np.abs(Ex - Ez)

    # Visualize
    print('Epoch: {}; m: {:.4}, L: {:.4}'.format(t, margin, L))

    samples = sess.run(G_sample, feed_dict={z: GET_NOISE(16, NOISE_SIZE)})

    fig = plot(samples)
    plt.savefig(SAVE_DIR + '/{}.png'
                .format(str(i).zfill(3)), bbox_inches='tight')
    i += 1
    plt.close(fig)
