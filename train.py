'''
CS7180 - Watermark removal for natural images using GAN.
Author - Amit Mulay

Execution instruction on discovery HPC node:
Required Modules:
CUDA 10.2
Python 3.7
Tensorflow-gpu
Discovery HPC partition required - gpu size - 128 GB time:08:00:00 (For 50 epochs)
'''

from tensorflow import keras
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
import scipy.misc
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import math
import random
import os.path

#For
X_train_w = np.load("/scratch/mulay.am/datasets/CLWD_1/X_train_w_10k.npy")
X_train_g = np.load("/scratch/mulay.am/datasets/CLWD_1/X_train_g_10k.npy")
X_test_w  = np.load("/scratch/mulay.am/datasets/CLWD_1/X_test_w_5.npy")
X_test_g  = np.load("/scratch/mulay.am/datasets/CLWD_1/X_test_g_5.npy")

dir_result = "/scratch/mulay.am/1watermark/Results/5/"
input_size = (256, 256, 3)

def get_generator_model(pretrained_weights=None, input_size=input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7])

    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8])

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))

    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


def get_optimizer():
    return Adam(lr=1e-4)


def get_discriminator_model(input_size=input_size):
    def build_layer(layer_input, filters, f_size=4, bn=True):
        layer = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        layer = LeakyReLU(alpha=0.2)(layer)
        if bn:
            layer = BatchNormalization(momentum=0.8)(layer)
        return layer

    img_A = Input(input_size)
    img_B = Input(input_size)

    number_of_filters = 64

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = build_layer(combined_imgs, number_of_filters, bn=False)
    d2 = build_layer(d1, number_of_filters * 2)
    d3 = build_layer(d2, number_of_filters * 4)
    d4 = build_layer(d3, number_of_filters * 4)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)

    discriminator = Model([img_A, img_B], validity)
    discriminator.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    return discriminator

def train(generator, discriminator, X_train_w, X_train_g, epochs=1, batch_size=4):
    history = []
    adam = get_optimizer()
    gan = get_gan_network(discriminator, generator, adam)
    for epoch in range(epochs):
        print('epoch',epoch)

        wat_batch = X_train_w
        gt_batch = X_train_g
        batch_count = wat_batch.shape[0] // batch_size
        for b in (range(batch_count)):

            seed = range(b * batch_size, (b * batch_size) + batch_size)

            b_wat_batch = wat_batch[seed].reshape(batch_size, 256, 256, 3)
            b_gt_batch = gt_batch[seed].reshape(batch_size, 256, 256, 3)
            generated_images = generator.predict(b_wat_batch)
            #print('shape',generated_images.shape)
            valid = np.ones((b_gt_batch.shape[0],) + (16, 16, 1))
            fake = np.zeros((b_gt_batch.shape[0],) + (16, 16, 1))

            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch([b_gt_batch, b_wat_batch], valid)
            d_loss_fake = discriminator.train_on_batch([generated_images, b_wat_batch], fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            discriminator.trainable = False
            g_loss = gan.train_on_batch([b_wat_batch], [valid, b_gt_batch])
            history.append({"D": d_loss[0], "G": g_loss})
        if epoch % 10 == 0:
            plot_generated_images(generator,X_test_w,
                              path_save=dir_result + "/image_{:05.0f}.png".format(epoch),
                              titleadd="Epoch {}".format(epoch))
            gan.save(dir_result)


def get_gan_network(discriminator, generator, optimizer, input_size=input_size):
    discriminator.trainable = False

    gan_input2 = Input(input_size)

    x = generator(gan_input2)
    valid = discriminator([x, gan_input2])
    gan = Model(inputs=[gan_input2], outputs=[valid, x])
    gan.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[1, 100], optimizer=optimizer, metrics=['accuracy'])
    return gan



def plot_generated_images(generator, noise, path_save=None, titleadd=""):
    imgs = generator.predict(X_test_w)
    fig = plt.figure(figsize=(40, 10))
    nsample = X_test_w.shape[0]
    for i, img in enumerate(imgs):

        ax = fig.add_subplot(1, nsample, i + 1)
        ax.imshow(img)

    fig.suptitle("Generated images " + titleadd, fontsize=30)

    if path_save is not None:
        plt.savefig(path_save,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()
    else:
        plt.close()

generator = get_generator_model()
discriminator = get_discriminator_model()
batch_size = 4
epochs = 51
train(generator, discriminator, X_train_w, X_train_g, epochs, batch_size)

