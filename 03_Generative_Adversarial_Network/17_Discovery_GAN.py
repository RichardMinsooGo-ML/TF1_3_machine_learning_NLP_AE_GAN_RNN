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

SAVE_DIR = "/tmp/ML/13_Generative_Adversarial_Network/17_Disco_GAN"

# Define Hyper Parameters
N_EPISODES = 10000
mb_size = 32
INPUT_SIZE = 784
NOISE_SIZE = 128
H_SIZE_01 = 256
lr = 1e-3
d_steps = 3

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

def log(x):
    return tf.log(x + 1e-8)

X_A = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
X_B = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])

"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Generator Variables
W01_Gen_AB = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Gen_AB = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_Gen_AB = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Gen_AB = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))

W01_Gen_BA = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Gen_BA = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_Gen_BA = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Gen_BA = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))

# Discriminator Variables
W01_Dis_A  = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Dis_A  = tf.Variable(xavier_init([H_SIZE_01, 1]))
B01_Dis_A  = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Dis_A  = tf.Variable(tf.zeros(shape=[1]))

W01_Dis_B  = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Dis_B  = tf.Variable(xavier_init([H_SIZE_01, 1]))
B01_Dis_B  = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Dis_B  = tf.Variable(tf.zeros(shape=[1]))
"""

W01_Gen_AB = tf.get_variable("W01_Gen_AB", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Gen_AB = tf.get_variable("W02_Gen_AB", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Gen_AB = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Gen_AB = tf.Variable(tf.random_normal([INPUT_SIZE]))

W01_Gen_BA = tf.get_variable("W01_Gen_BA", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Gen_BA = tf.get_variable("W02_Gen_BA", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Gen_BA = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Gen_BA = tf.Variable(tf.random_normal([INPUT_SIZE]))

W01_Dis_A  = tf.get_variable("W01_Dis_A", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis_A  = tf.get_variable("W02_Dis_A", shape=[H_SIZE_01, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Dis_A  = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis_A  = tf.Variable(tf.random_normal([1]))

W01_Dis_B  = tf.get_variable("W01_Dis_B", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis_B  = tf.get_variable("W02_Dis_B", shape=[H_SIZE_01, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Dis_B  = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis_B  = tf.Variable(tf.random_normal([1]))

# Build Generator Network.
def GENERATOR_AtoB(X):
    _LAY01_Gen_AtoB = tf.nn.relu(tf.matmul(X, W01_Gen_AB) + B01_Gen_AB)
    output_Gen_AtoB = tf.nn.sigmoid(tf.matmul(_LAY01_Gen_AtoB, W02_Gen_AB) + B02_Gen_AB)
    return output_Gen_AtoB

def GENERATOR_BtoA(X):
    _LAY01_Gen_BtoA = tf.nn.relu(tf.matmul(X, W01_Gen_BA) + B01_Gen_BA)
    output_Gen_BtoA = tf.nn.sigmoid(tf.matmul(_LAY01_Gen_BtoA, W02_Gen_BA) + B02_Gen_BA)
    return output_Gen_BtoA

# Build Discriminator Network
def DISCRIMINATOR_A(X):
    _LAY01_Dis_A = tf.nn.relu(tf.matmul(X, W01_Dis_A) + B01_Dis_A)
    output_Dis_A = tf.nn.sigmoid(tf.matmul(_LAY01_Dis_A, W02_Dis_A) + B02_Dis_A)
    return output_Dis_A

def DISCRIMINATOR_B(X):
    _LAY01_Dis_B = tf.nn.relu(tf.matmul(X, W01_Dis_B) + B01_Dis_B)
    output_Dis_B = tf.nn.sigmoid(tf.matmul(_LAY01_Dis_B, W02_Dis_B) + B02_Dis_B)
    return output_Dis_B

G_var_list = [W01_Gen_AB, W02_Gen_AB, B01_Gen_AB, B02_Gen_AB,
           W01_Gen_BA, W02_Gen_BA, B01_Gen_BA, B02_Gen_BA]
D_var_list = [W01_Dis_A, W02_Dis_A, B01_Dis_A, B02_Dis_A,
           W01_Dis_B, W02_Dis_B, B01_Dis_B, B02_Dis_B]

# Discriminator A
X_BA = GENERATOR_BtoA(X_B)
Dis_A_real = DISCRIMINATOR_A(X_A)
Dis_A_fake = DISCRIMINATOR_A(X_BA)

# Discriminator B
X_AB = GENERATOR_AtoB(X_A)
Dis_B_real = DISCRIMINATOR_B(X_B)
Dis_B_fake = DISCRIMINATOR_B(X_AB)

# Generator AB
X_ABA = GENERATOR_BtoA(X_AB)

# Generator BA
X_BAB = GENERATOR_AtoB(X_BA)

# Discriminator loss
Loss_Dis_A = -tf.reduce_mean(log(Dis_A_real) + log(1 - Dis_A_fake))
Loss_Dis_B = -tf.reduce_mean(log(Dis_B_real) + log(1 - Dis_B_fake))

D_loss = Loss_Dis_A + Loss_Dis_B

# Generator loss
L_adv_B = -tf.reduce_mean(log(Dis_B_fake))
L_recon_A = tf.reduce_mean(tf.reduce_sum((X_A - X_ABA)**2, 1))
Loss_Gen_AtoB = L_adv_B + L_recon_A

L_adv_A = -tf.reduce_mean(log(Dis_A_fake))
L_recon_B = tf.reduce_mean(tf.reduce_sum((X_B - X_BAB)**2, 1))
Loss_Gen_BtoA = L_adv_A + L_recon_B

G_loss = Loss_Gen_AtoB + Loss_Gen_BtoA

# Solvers
solver = tf.train.AdamOptimizer(learning_rate=lr)
D_solver = solver.minimize(D_loss, var_list=D_var_list)
G_solver = solver.minimize(G_loss, var_list=G_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Gather training data from 2 domains
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

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

i = 0

for episode in range(N_EPISODES):
    # Sample data from both domains
    X_A_mb = sample_X(X_train1, mb_size)
    X_B_mb = sample_X(X_train2, mb_size)

    _, D_loss_curr = sess.run(
        [D_solver, D_loss], feed_dict={X_A: X_A_mb, X_B: X_B_mb}
    )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss], feed_dict={X_A: X_A_mb, X_B: X_B_mb}
    )

    if episode % 1000 == 0:
        print("[Episode : {:>5d}] [D_loss: {:2.5f}] [G_loss: {:2.5f}]"
              .format(episode, D_loss_curr, G_loss_curr))

        input_A = sample_X(X_train1, size=4)
        input_B = sample_X(X_train2, size=4)

        samples_A = sess.run(X_BA, feed_dict={X_B: input_B})
        samples_B = sess.run(X_AB, feed_dict={X_A: input_A})

        # The resulting image sample would be in 4 rows:
        # row 1: real data from domain A, row 2 is its domain B translation
        # row 3: real data from domain B, row 4 is its domain A translation
        samples = np.vstack([input_A, samples_B, input_B, samples_A])

        fig = plot(samples)
        plt.savefig(SAVE_DIR + '/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
