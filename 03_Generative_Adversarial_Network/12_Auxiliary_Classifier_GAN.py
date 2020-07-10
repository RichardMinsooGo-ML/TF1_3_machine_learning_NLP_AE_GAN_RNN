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

SAVE_DIR = "/tmp/ML/13_Generative_Adversarial_Network/12_Auxiliary_Classifier"

# Define Hyper Parameters
N_EPISODES = 10000
mb_size = 32

INPUT_SIZE = mnist.train.images.shape[1]
# INPUT_SIZE = 784
OUTPUT_SIZE = mnist.train.labels.shape[1]
NOISE_SIZE = 128
H_SIZE_01 = 256
eps = 1e-8
lr = 1e-3
d_steps = 3
N_SAMPLE = 16

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
y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
z = tf.placeholder(tf.float32, shape=[None, NOISE_SIZE])

"""
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

# Generator Variables
W01_Gen     = tf.Variable(xavier_init([NOISE_SIZE + OUTPUT_SIZE, H_SIZE_01]))
W02_Gen     = tf.Variable(xavier_init([H_SIZE_01, INPUT_SIZE]))
B01_Gen     = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Gen     = tf.Variable(tf.zeros(shape=[INPUT_SIZE]))

# Discriminator Variable
W01_Dis     = tf.Variable(xavier_init([INPUT_SIZE, H_SIZE_01]))
W02_Dis_gan = tf.Variable(xavier_init([H_SIZE_01, 1]))
W02_Dis_aux = tf.Variable(xavier_init([H_SIZE_01, OUTPUT_SIZE]))
B01_Dis     = tf.Variable(tf.zeros(shape=[H_SIZE_01]))
B02_Dis_gan = tf.Variable(tf.zeros(shape=[1]))
B02_Dis_aux = tf.Variable(tf.zeros(shape=[OUTPUT_SIZE]))
"""

W01_Gen     = tf.get_variable("W01_Gen", shape=[NOISE_SIZE + OUTPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Gen     = tf.get_variable("W02_Gen", shape=[H_SIZE_01, INPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Gen     = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Gen     = tf.Variable(tf.random_normal([INPUT_SIZE]))

W01_Dis     = tf.get_variable("W01_Dis", shape=[INPUT_SIZE, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis_gan = tf.get_variable("W02_Dis_gan", shape=[H_SIZE_01, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
W02_Dis_aux = tf.get_variable("W02_Dis_aux", shape=[H_SIZE_01, OUTPUT_SIZE],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_Dis     = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis_gan = tf.Variable(tf.random_normal([1]))
B02_Dis_aux = tf.Variable(tf.random_normal([OUTPUT_SIZE]))

# Build Generator Network.
def GENERATOR(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    _LAY01_Gen = tf.nn.relu(tf.matmul(inputs, W01_Gen) + B01_Gen)
    output_gen = tf.nn.sigmoid(tf.matmul(_LAY01_Gen, W02_Gen) + B02_Gen)
    return output_gen

def DISCRIMINATOR(X):
    _LAY01_Dis = tf.nn.relu(tf.matmul(X, W01_Dis) + B01_Dis)
    output_gan = tf.nn.sigmoid(tf.matmul(_LAY01_Dis, W02_Dis_gan) + B02_Dis_gan)
    output_aux = tf.matmul(_LAY01_Dis, W02_Dis_aux) + B02_Dis_aux
    return output_gan, output_aux

def cross_entropy(logit, y):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

def GET_NOISE(BATCH_SIZE, NOISE_SIZE):
#    return np.random.uniform(-1., 1., size=[BATCH_SIZE, NOISE_SIZE])
    return np.random.normal(-1., 1., size=[BATCH_SIZE, NOISE_SIZE])

G_var_list = [W01_Gen, W02_Gen, B01_Gen, B02_Gen]
D_var_list = [W01_Dis, W02_Dis_gan, W02_Dis_aux, B01_Dis, B02_Dis_gan, B02_Dis_aux]

G_sample = GENERATOR(z, y)

D_real, C_real = DISCRIMINATOR(X)
D_fake, C_fake = DISCRIMINATOR(G_sample)

# Cross entropy aux loss
C_loss = cross_entropy(C_real, y) + cross_entropy(C_fake, y)

# GAN D loss
D_loss = tf.reduce_mean(tf.log(D_real + eps) + tf.log(1. - D_fake + eps))
DC_loss = -(D_loss + C_loss)

# GAN's G loss
G_loss = tf.reduce_mean(tf.log(D_fake + eps))
GC_loss = -(G_loss + C_loss)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(DC_loss, var_list=D_var_list))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(GC_loss, var_list=G_var_list))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    
i = 0

for episode in range(N_EPISODES):
    X_mb, y_mb = mnist.train.next_batch(mb_size)
    z_mb = GET_NOISE(mb_size, NOISE_SIZE)

    _, DC_loss_curr = sess.run(
        [D_solver, DC_loss],
        feed_dict={X: X_mb, y: y_mb, z: z_mb}
    )

    _, GC_loss_curr = sess.run(
        [G_solver, GC_loss],
        feed_dict={X: X_mb, y: y_mb, z: z_mb}
    )

    if episode % 1000 == 0:
        idx = np.random.randint(0, 10)
        c = np.zeros([N_SAMPLE, OUTPUT_SIZE])
        c[range(N_SAMPLE), idx] = 1

        samples = sess.run(G_sample, feed_dict={z: GET_NOISE(N_SAMPLE, NOISE_SIZE), y: c})

        print('[Episode : {:>5}] [DC_loss: {:2.5f}] [GC_loss: {:2.5f}] [Idx; {}]'
              .format(episode, DC_loss_curr, GC_loss_curr, idx))

        fig = plot(samples)
        plt.savefig(SAVE_DIR + '/{}.png'
                    .format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
