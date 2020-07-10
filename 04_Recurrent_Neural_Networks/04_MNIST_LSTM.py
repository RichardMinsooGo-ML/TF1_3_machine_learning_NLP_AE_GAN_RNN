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

#define constants

#hidden LSTM units
H_SIZE_01 = 128

INPUT_SIZE = 28           # rows of 28 pixels
TIME_STEP = 28            # unrolled through 28 time steps
learning_rate=0.001     # learning rate for adam
#mnist is meant to be classified in 10 classes(0-9).
n_classes=10
#size of batch
batch_size=128

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([H_SIZE_01,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x = tf.placeholder("float",[None,TIME_STEP,INPUT_SIZE])
#input label placeholder
y = tf.placeholder("float",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,INPUT_SIZE] to "TIME_STEP" number of [batch_size,INPUT_SIZE] tensors
input=tf.unstack(x ,TIME_STEP,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(H_SIZE_01,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,H_SIZE_01] to [batch_size,n_classes] by out_weight multiplication
Pred_m=tf.matmul(outputs[-1],out_weights)+out_bias

#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_m,labels=y))
#optimization
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(Pred_m,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter<800:
        batch_x, batch_y=mnist.train.next_batch(batch_size=batch_size)

        batch_x=batch_x.reshape((batch_size,TIME_STEP,INPUT_SIZE))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1


    #calculating test accuracy
    test_data = mnist.test.images[:128].reshape((-1, TIME_STEP, INPUT_SIZE))
    test_label = mnist.test.labels[:128]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


    #########
    # 결과 확인 (matplot)
    ######
    labels = sess.run(Pred_m,
                      feed_dict={x: mnist.test.images.reshape(-1, 28, 28),
                                 y: mnist.test.labels})

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

