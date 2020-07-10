from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

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
LOG_DIR_Model = "/tmp/ML/13_Generative_Adversarial_Network/BidirectionalGAN/Model"
LOG_DIR_Graph = "/tmp/ML/13_Generative_Adversarial_Network/BidirectionalGAN/Graph"

Print_Cycle = 20

class BIGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer)


    def build_encoder(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        model.summary()

        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, img], validity)

    def train(self, N_EPISODES, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for episode in range(N_EPISODES):


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img
            z = np.random.normal(size=(batch_size, self.latent_dim))
            imgs_ = self.generator.predict(z)

            # Select a random batch of images and encode
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            z_ = self.encoder.predict(imgs)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = self.bigan_generator.train_on_batch([z, imgs], [valid, fake])

            # Plot the progress
            if (episode+1) % Print_Cycle == 0:
                print ("[Episodes: %6d] [D loss: %3.6f, acc.: %.2f%%] [G loss: %3.6f]" % (episode+1, d_loss[0], 100*d_loss[1], g_loss[0]))

            # If at save interval => save generated image samples
            if (episode+1) % sample_interval == 0:
                self.save_model()
                self.sample_interval(episode)

    def sample_interval(self, episode):
        _row, _column = 5, 5
        z = np.random.normal(size=(25, self.latent_dim))
        gen_imgs = self.generator.predict(z)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(_row, _column)
        _count = 0
        for i in range(_row):
            for j in range(_column):
                axs[i,j].imshow(gen_imgs[_count,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                _count += 1
        fig.savefig(LOG_DIR_Graph + "/%d.png" % (episode+1))
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
        
    bigan = BIGAN()
    bigan.train(N_EPISODES=5000, batch_size=32, sample_interval=200)
