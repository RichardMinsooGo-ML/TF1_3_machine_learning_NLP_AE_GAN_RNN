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

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
DATA_DIR = "/tmp/ML/MNIST_data"
mnist=input_data.read_data_sets(DATA_DIR,one_hot=True)

###Neural Network Parameters
### 28*28=784

num_inputs = 784 

num_h1 = 256 
num_h2 = 128

###Tensorflow data inputs

X=tf.placeholder("float",shape=[None,num_inputs])

### Neural Network training parameters

batch_size=100
N_EPISODES=5000
learning_rate=0.001
display_episode=100

###Creating Network Architechture

def encoder_layer(x):
    l1=tf.matmul(x,W["w1"])
    l1=tf.add(l1,b["b1"])
    l1=tf.nn.sigmoid(l1)
    
    l1=tf.matmul(l1,W["w2"])
    l1=tf.add(l1,b["b2"])
    l1=tf.nn.sigmoid(l1)
    
    return l1
    
    

def decoder_layer(x):
    l2=tf.matmul(x,W["w3"])
    l2=tf.add(l2,b["b3"])
    l2=tf.nn.sigmoid(l2)
    
    l2=tf.matmul(l2,W["w4"])
    l2=tf.add(l2,b["b4"])
    l2=tf.nn.sigmoid(l2)
    
    return l2

###Model Architechture Weights


W={"w1":tf.Variable(tf.random_normal([num_inputs,num_h1])),
   "w2": tf.Variable(tf.random_normal([num_h1,num_h2])),
   "w3": tf.Variable(tf.random_normal([num_h2,num_h1])), 
   "w4": tf.Variable(tf.random_normal([num_h1,num_inputs]))} 

b={"b1":tf.Variable(tf.random_normal([num_h1])),
   "b2":tf.Variable(tf.random_normal([num_h2])),
   "b3":tf.Variable(tf.random_normal([num_h1])),
   "b4":tf.Variable(tf.random_normal([num_inputs]))}

###Model Architechture

encoder_fun = encoder_layer(X)
decoder_fun = decoder_layer(encoder_fun)

####Cost function Evaluation


# Prediction
predicted = decoder_fun

#Actual
actual=X

cost_fn=tf.reduce_mean(tf.pow(actual - predicted, 2))
optim=tf.train.RMSPropOptimizer(learning_rate=learning_rate)
training=optim.minimize(cost_fn)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

###Staring the Model training Session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for episode in range(1, N_EPISODES+1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(training, feed_dict={X: batch_x})
        if episode % display_episode == 0 or episode == 1:
            # Calculate batch loss and accuracy
            loss, _ = sess.run([cost_fn, training], feed_dict={X: batch_x})
            print("Episode " + str(episode) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss))

    print("Optimization Finished!")
    
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))


    for i in range(n):

        batch_x, _ = mnist.test.next_batch(n)

        # Session 
        g = sess.run(decoder_fun, feed_dict={X: batch_x})
    
        # original images
        for j in range(n):
            # Draw the generated digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
    
        #  reconstructed images
        for j in range(n):
            # Draw the generated digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    print("Original Images")     
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

