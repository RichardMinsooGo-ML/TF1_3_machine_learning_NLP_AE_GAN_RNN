import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import scipy.ndimage.interpolation

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

SAVE_DIR = "/tmp/ML/13_Generative_Adversarial_Network/18_Dual_GAN"

# Define Hyper Parameters
N_EPISODES = 10000
mb_size = 32

# Parameters for the Networks
INPUT_SIZE = mnist.train.images.shape[1]
OUTPUT_SIZE = mnist.train.labels.shape[1]
NOISE_SIZE = 128
H_SIZE_01 = 256
eps = 1e-8
lr = 1e-3
d_steps = 3
lam1, lam2 = 1000, 1000

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

X1 = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
X2 = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE])

"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Generator Variables
W01_Gen_01 = tf.Variable(xavier_init([INPUT_SIZE + NOISE_SIZE, H_SIZE_01]))
W02_Gen_01 = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_Gen_01 = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Gen_01 = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))

W01_Gen_02 = tf.Variable(xavier_init([INPUT_SIZE + NOISE_SIZE, H_SIZE_01]))
W02_Gen_02 = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_Gen_02 = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Gen_02 = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))

# Discriminator Variables
W01_Dis_01 = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Dis_01 = tf.Variable(xavier_init([H_SIZE_01, 1]))
B01_Dis_01 = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Dis_01 = tf.Variable(tf.zeros(shape=[1]))

W01_Dis_02 = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Dis_02 = tf.Variable(xavier_init([H_SIZE_01, 1]))
B01_Dis_02 = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Dis_02 = tf.Variable(tf.zeros(shape=[1]))
"""

W01_Gen_01 = tf.get_variable("W01_Gen_01", shape=[INPUT_SIZE + NOISE_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Gen_01 = tf.get_variable("W02_Gen_01", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Gen_01 = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Gen_01 = tf.Variable(tf.random_normal([INPUT_SIZE]))

W01_Gen_02 = tf.get_variable("W01_Gen_02", shape=[INPUT_SIZE + NOISE_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Gen_02 = tf.get_variable("W02_Gen_02", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Gen_02 = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Gen_02 = tf.Variable(tf.random_normal([INPUT_SIZE]))

W01_Dis_01 = tf.get_variable("W01_Dis_01", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis_01 = tf.get_variable("W02_Dis_01", shape=[H_SIZE_01, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Dis_01 = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis_01 = tf.Variable(tf.random_normal([1]))

W01_Dis_02 = tf.get_variable("W01_Dis_02", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis_02 = tf.get_variable("W02_Dis_02", shape=[H_SIZE_01, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Dis_02 = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis_02 = tf.Variable(tf.random_normal([1]))

# Build Generator Network.
def GENERATOR_01(X1, z):
    inputs = tf.concat([X1, z], 1)
    _LAY01_Gen1 = tf.nn.relu(tf.matmul(inputs, W01_Gen_01) + B01_Gen_01)
    output_Gen1 = tf.nn.sigmoid(tf.matmul(_LAY01_Gen1, W02_Gen_01) + B02_Gen_01)
    return output_Gen1

def GENERATOR_02(X2, z):
    inputs = tf.concat([X2, z], 1)
    _LAY01_Gen2 = tf.nn.relu(tf.matmul(inputs, W01_Gen_02) + B01_Gen_02)
    output_Gen2 = tf.nn.sigmoid(tf.matmul(_LAY01_Gen2, W02_Gen_02) + B02_Gen_02)
    return output_Gen2

def DISCRIMINATOR_01(X):
    _LAY01_Dis1 = tf.nn.relu(tf.matmul(X, W01_Dis_01) + B01_Dis_01)
    output_Dis1 = tf.matmul(_LAY01_Dis1, W02_Dis_01) + B02_Dis_01
    return output_Dis1

def DISCRIMINATOR_02(X):
    _LAY01_Dis2 = tf.nn.relu(tf.matmul(X, W01_Dis_01) + B01_Dis_01)
    output_Dis2 = tf.matmul(_LAY01_Dis2, W02_Dis_02) + B02_Dis_02
    return output_Dis2

G_01_var_list = [W01_Gen_01, W02_Gen_01, B02_Gen_01, B02_Gen_01]
G_02_var_list = [W01_Gen_02, B01_Gen_02, W02_Gen_02, B02_Gen_02]
G_var_list = G_01_var_list + G_02_var_list

D_01_var_list = [W01_Dis_01, W02_Dis_01, B01_Dis_01, B02_Dis_01]
D_02_var_list = [W01_Dis_02, B01_Dis_02, W02_Dis_02, B02_Dis_02]

# D
X1_sample = GENERATOR_02(X2, z)
X2_sample = GENERATOR_01(X1, z)

D_01_real = DISCRIMINATOR_01(X2)
D_01_fake = DISCRIMINATOR_01(X2_sample)

D_02_real = DISCRIMINATOR_02(X1)
D_02_fake = DISCRIMINATOR_02(X1_sample)

D_01_Gen = DISCRIMINATOR_01(X1_sample)
D_02_Gen = DISCRIMINATOR_02(X2_sample)

X1_recon = GENERATOR_02(X2_sample, z)
X2_recon = GENERATOR_01(X1_sample, z)
recon1 = tf.reduce_mean(tf.reduce_sum(tf.abs(X1 - X1_recon), 1))
recon2 = tf.reduce_mean(tf.reduce_sum(tf.abs(X2 - X2_recon), 1))

Loss_D_01 = tf.reduce_mean(D_01_fake) - tf.reduce_mean(D_01_real)
Loss_D_02 = tf.reduce_mean(D_02_fake) - tf.reduce_mean(D_02_real)
G_loss = -tf.reduce_mean(D_01_Gen + D_02_Gen) + lam1*recon1 + lam2*recon2

Train_D_01 = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
             .minimize(Loss_D_01, var_list=D_01_var_list))
Train_D_02 = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
             .minimize(Loss_D_02, var_list=D_02_var_list))
train_G = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=G_var_list))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in D_01_var_list + D_02_var_list]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

X_train = mnist.train.images
half = int(X_train.shape[0] / 2)

# Real image
X_train1 = X_train[:half]
# Rotated image
X_train2 = X_train[half:].reshape(-1, 28, 28)
X_train2 = scipy.ndimage.interpolation.rotate(X_train2, 90, axes=(1, 2))
X_train2 = X_train2.reshape(-1, 28*28)

# Cleanup
del X_train

def sample_X(X, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size]


def _GET_NOISE(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

i = 0

for episode in range(N_EPISODES):
    for _ in range(d_steps):
        X1_mb, X2_mb = sample_X(X_train1, mb_size), sample_X(X_train2, mb_size)
        z_mb = _GET_NOISE(mb_size, NOISE_SIZE)

        _, _, Loss_D_01_curr, Loss_D_02_curr, _ = sess.run(
            [Train_D_01, Train_D_02, Loss_D_01, Loss_D_02, clip_D],
            feed_dict={X1: X1_mb, X2: X2_mb, z: z_mb}
        )

    _, G_loss_curr = sess.run(
        [train_G, G_loss], feed_dict={X1: X1_mb, X2: X2_mb, z: z_mb}
    )

    if episode % 1000 == 0:
        sample1, sample2 = sess.run(
            [X1_sample, X2_sample],
            feed_dict={X1: X1_mb[:4], X2: X2_mb[:4], z: _GET_NOISE(4, NOISE_SIZE)}
        )

        samples = np.vstack([X1_mb[:4], sample1, X2_mb[:4], sample2])

        print("[Episode : {:>5d}] [D_loss: {:2.5f}] [G_loss: {:2.5f}]"
              .format(episode, Loss_D_01_curr + Loss_D_02_curr, G_loss_curr))

        fig = plot(samples)
        plt.savefig(SAVE_DIR + '/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
