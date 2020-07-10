from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
import sys
import scipy

import tensorflow as tf
import os
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 13.	(option) Model Save path define and Initialization
LOG_DIR_Model = "/tmp/ML/13_Generative_Adversarial_Network/Coupled_GAN/Model"
LOG_DIR_Graph = "/tmp/ML/13_Generative_Adversarial_Network/Coupled_GAN/Graph"

Print_Cycle = 20

class COGAN():
    
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.d1, self.d2 = self.build_discriminators()
        self.d1.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d2.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.g1, self.g2 = self.build_generators()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        image_1 = self.g1(z)
        image_2 = self.g2(z)

        # For the combined model we will only train the generators
        self.d1.trainable = False
        self.d2.trainable = False

        # The valid takes generated images as input and determines validity
        valid_1 = self.d1(image_1)
        valid_2 = self.d2(image_2)

        # The combined model  (stacked generators and discriminators)
        # Trains generators to fool discriminators
        self.combined = Model(z, [valid_1, valid_2])
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                                    optimizer=optimizer)

    def build_generators(self):

        # Shared weights between generators
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        noise = Input(shape=(self.latent_dim,))
        feature_repr = model(noise)

        # Generator 1
        g1 = Dense(1024)(feature_repr)
        g1 = LeakyReLU(alpha=0.2)(g1)
        g1 = BatchNormalization(momentum=0.8)(g1)
        g1 = Dense(np.prod(self.img_shape), activation='tanh')(g1)
        image_1 = Reshape(self.img_shape)(g1)

        # Generator 2
        g2 = Dense(1024)(feature_repr)
        g2 = LeakyReLU(alpha=0.2)(g2)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Dense(np.prod(self.img_shape), activation='tanh')(g2)
        image_2 = Reshape(self.img_shape)(g2)

        model.summary()

        return Model(noise, image_1), Model(noise, image_2)

    def build_discriminators(self):

        image_1 = Input(shape=self.img_shape)
        image_2 = Input(shape=self.img_shape)

        # Shared discriminator layers
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        image_1_embedding = model(image_1)
        image_2_embedding = model(image_2)

        # Discriminator 1
        validity1 = Dense(1, activation='sigmoid')(image_1_embedding)
        # Discriminator 2
        validity2 = Dense(1, activation='sigmoid')(image_2_embedding)

        return Model(image_1, validity1), Model(image_2, validity2)

    def train(self, N_EPISODES, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Images in domain A and B (rotated)
        X1 = X_train[:int(X_train.shape[0]/2)]
        X2 = X_train[int(X_train.shape[0]/2):]
        X2 = scipy.ndimage.interpolation.rotate(X2, 90, axes=(1, 2))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for episode in range(N_EPISODES):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X1.shape[0], batch_size)
            imgs1 = X1[idx]
            imgs2 = X2[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a batch of new images
            gen_imgs1 = self.g1.predict(noise)
            gen_imgs2 = self.g2.predict(noise)

            # Train the discriminators
            d1_loss_real = self.d1.train_on_batch(imgs1, valid)
            d2_loss_real = self.d2.train_on_batch(imgs2, valid)
            d1_loss_fake = self.d1.train_on_batch(gen_imgs1, fake)
            d2_loss_fake = self.d2.train_on_batch(gen_imgs2, fake)
            d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
            d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)


            # ------------------
            #  Train Generators
            # ------------------

            g_loss = self.combined.train_on_batch(noise, [valid, valid])

            # Plot the progress
            if (episode+1) % Print_Cycle == 0:
                print ("[Episodes: %6d] [D1 loss: %2.6f, acc.: %2.2f%%] [D2 loss: %2.6f, acc.: %2.2f%%] [G loss: %2.6f]" \
                    % (episode+1, d1_loss[0], 100*d1_loss[1], d2_loss[0], 100*d2_loss[1], g_loss[0]))

            # If at save interval => save generated image samples
            if (episode+1) % sample_interval == 0:
                self.sample_images(episode)

    def sample_images(self, episode):
        _row, _colimn = 4, 4
        noise = np.random.normal(0, 1, (_row * int(_colimn/2), 100))
        gen_imgs1 = self.g1.predict(noise)
        gen_imgs2 = self.g2.predict(noise)

        gen_imgs = np.concatenate([gen_imgs1, gen_imgs2])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(_row, _colimn)
        _count = 0
        for i in range(_row):
            for j in range(_colimn):
                axs[i,j].imshow(gen_imgs[_count, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                _count += 1
        fig.savefig(LOG_DIR_Graph + "/%d.png" % episode)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = LOG_DIR_Model + "/%s.json" % model_name
            weights_path = LOG_DIR_Model + "/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    
    if not os.path.exists(LOG_DIR_Model):
        os.makedirs(LOG_DIR_Model)
    if not os.path.exists(LOG_DIR_Graph):
        os.makedirs(LOG_DIR_Graph)
        
    gan = COGAN()
    gan.train(N_EPISODES=5000, batch_size=32, sample_interval=200)
