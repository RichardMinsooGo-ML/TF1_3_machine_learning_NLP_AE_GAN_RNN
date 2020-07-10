import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
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

SAVE_DIR = "/tmp/ML/13_Generative_Adversarial_Network/01_Simple_GAN"
    
#########
# 옵션 설정
######
N_EPISODES = 100
batch_size = 100
learning_rate = 0.0002
# 신경망 레이어 구성 옵션
H_SIZE_01 = 256
INPUT_SIZE = 28 * 28
NOISE_SIZE = 128  # 생성기의 입력값으로 사용할 노이즈의 크기

#########
# 신경망 모델 구성
######
# GAN 도 Unsupervised 학습이므로 Autoencoder 처럼 Y 를 사용하지 않습니다.
X = tf.placeholder(tf.float32, [None, INPUT_SIZE])
# 노이즈 Z를 입력값으로 사용합니다.
Z = tf.placeholder(tf.float32, [None, NOISE_SIZE])

"""
# 생성기 신경망에 사용하는 변수들입니다.
W01_Gen = tf.Variable(tf.random_normal([NOISE_SIZE, H_SIZE_01], stddev=0.01))
W02_Gen = tf.Variable(tf.random_normal([H_SIZE_01, INPUT_SIZE], stddev=0.01))
B01_Gen = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Gen = tf.Variable(tf.random_normal([INPUT_SIZE]))

# 판별기 신경망에 사용하는 변수들입니다.
W01_Dis = tf.Variable(tf.random_normal([INPUT_SIZE, H_SIZE_01], stddev=0.01))
W02_Dis = tf.Variable(tf.random_normal([H_SIZE_01, 1], stddev=0.01))
B01_Dis = tf.Variable(tf.random_normal([H_SIZE_01]))
B02_Dis = tf.Variable(tf.random_normal([1]))
"""
W01_Gen = tf.get_variable("W01_Gen",shape = [NOISE_SIZE, H_SIZE_01]
                          ,initializer = tf.contrib.layers.xavier_initializer() )
W02_Gen = tf.get_variable("W02_Gen",shape = [H_SIZE_01, INPUT_SIZE]
                          ,initializer = tf.contrib.layers.xavier_initializer() )
B01_Gen = tf.Variable(tf.zeros([H_SIZE_01]))
B02_Gen = tf.Variable(tf.zeros([INPUT_SIZE]))

W01_Dis = tf.get_variable("W01_DIS",shape = [INPUT_SIZE, H_SIZE_01]
                          ,initializer = tf.contrib.layers.xavier_initializer() )
W02_Dis = tf.get_variable("W02_DIS",shape = [H_SIZE_01, 1]
                          ,initializer = tf.contrib.layers.xavier_initializer() )
B01_Dis = tf.Variable(tf.zeros([H_SIZE_01]))
B02_Dis = tf.Variable(tf.zeros([1]))

# 생성기(G) 신경망을 구성합니다.
def _GENERATOR(noise_z):
    hidden = tf.nn.relu(
                    tf.matmul(noise_z, W01_Gen) + B01_Gen)
    output = tf.nn.sigmoid(
                    tf.matmul(hidden, W02_Gen) + B02_Gen)

    return output

# 판별기(D) 신경망을 구성합니다.
def _DISCRIMINATOR(inputs):
    hidden = tf.nn.relu(
                    tf.matmul(inputs, W01_Dis) + B01_Dis)
    output = tf.nn.sigmoid(
                    tf.matmul(hidden, W02_Dis) + B02_Dis)

    return output


# 랜덤한 노이즈(Z)를 만듭니다.
def _GET_NOISE(batch_size, NOISE_SIZE):
    return np.random.normal(size=(batch_size, NOISE_SIZE))


# 노이즈를 이용해 랜덤한 이미지를 생성합니다.
G = _GENERATOR(Z)
# 노이즈를 이용해 생성한 이미지가 진짜 이미지인지 판별한 값을 구합니다.
D_gene = _DISCRIMINATOR(G)
# 진짜 이미지를 이용해 판별한 값을 구합니다.
D_real = _DISCRIMINATOR(X)

# .
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

# .
loss_G = tf.reduce_mean(tf.log(D_gene))

# loss_D 를 구할 때는 판별기 신경망에 사용되는 변수만 사용하고,
# loss_G 를 구할 때는 생성기 신경망에 사용되는 변수만 사용하여 최적화를 합니다.
D_var_list = [W01_Dis, B01_Dis, W02_Dis, B01_Dis]
G_var_list = [W01_Gen, B01_Gen, W02_Gen, B02_Gen]

# GAN 논문의 수식에 따르면 loss 를 극대화 해야하지만, minimize 하는 최적화 함수를 사용하기 때문에
# 최적화 하려는 loss_D 와 loss_G 에 음수 부호를 붙여줍니다.
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D,
                                                         var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G,
                                                         var_list=G_var_list)

#########
# 신경망 모델 학습
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
        
for episode in range(N_EPISODES):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = _GET_NOISE(batch_size, NOISE_SIZE)

        # 판별기와 생성기 신경망을 각각 학습시킵니다.
        _, loss_val_D = sess.run([train_D, loss_D],
                                 feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G],
                                 feed_dict={Z: noise})

    print("[Episode : %05d]" % int(episode + 1),
          "[D loss: {:2.5f}]".format(loss_val_D),
          "[G loss: {:2.5f}]".format(loss_val_G))

    #########
    # 학습이 되어가는 모습을 보기 위해 주기적으로 이미지를 생성하여 저장
    ######
    if episode == 0 or (episode + 1) % 10 == 0:
        sample_size = 10
        noise = _GET_NOISE(sample_size, NOISE_SIZE)
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig(SAVE_DIR + '/{}.png'.format(str(episode).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('Optimization Completed!')
