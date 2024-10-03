import os

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Cropping2D, Dropout, Input, LeakyReLU,
                                     ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.disable_v2_behavior()

print("Tensorflow version " + tf.__version__)
print("Devices:")
print(tf.config.list_physical_devices())

# configure os environment
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CH_AXIS = -1

# configure tensorflow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

# configure keras
K.set_image_data_format("channels_last")

CH_AXIS = -1
ISIZE = 1024  # height of the image
NC_IN = 1  # number of input channels (1 for greyscale, 3 for RGB)
NC_OUT = 1  # number of output channels (1 for greyscale, 3 for RGB)
# max layers in the discriminator not including sigmoid activation:
# 1 for 16, 2 for 34, 3 for 70, 4 for 142, and 5 for 286 (receptive field size)
MAX_LAYERS = 3

# generates tensors with a normal distribution with (mean, standard deviation)
# this is used as a matrix of weights
CONV_INIT = RandomNormal(0, 0.02)
GAMMA_INIT = RandomNormal(1.0, 0.02)


# The loss function
def LOSS_FN(OUTPUT, TARGET):
    return -K.mean(
        K.log(OUTPUT + 1e-12) * TARGET + K.log(1 - OUTPUT + 1e-12) * (1 - TARGET)
    )


# create a convolutional layer with f filters, and arguments a and k
def DN_CONV(f, *a, **k):
    return Conv2D(f, kernel_initializer=CONV_INIT, *a, **k)


# create a deconvolutional layer with f filters, and arguments a and k
def UP_CONV(f, *a, **k):
    return Conv2DTranspose(f, kernel_initializer=CONV_INIT, *a, **k)


# applies normalisation such that max is 1, and minimum is 0
def BATNORM():
    return BatchNormalization(
        momentum=0.9, axis=CH_AXIS, epsilon=1.01e-5, gamma_initializer=GAMMA_INIT
    )


# leaky ReLU (y = alpha*x for x < 0, y = x for x > 0)
def LEAKY_RELU(alpha):
    return LeakyReLU(alpha)


#  the descriminator
def BASIC_D(ISIZE, NC_IN, NC_OUT, MAX_LAYERS, kernel):
    # combines the inputs from the generator and the desired input
    INPUT_A, INPUT_B = Input(shape=(ISIZE, ISIZE, NC_IN)), Input(
        shape=(ISIZE, ISIZE, NC_OUT)
    )

    INPUT = Concatenate(axis=CH_AXIS)([INPUT_A, INPUT_B])

    if MAX_LAYERS == 0:
        N_FEATURE = 1  # number of filters to use
        # apply sigmoid activation
        L = DN_CONV(
            N_FEATURE, kernel_size=kernel, padding="same", activation="sigmoid"
        )(INPUT)

    else:
        N_FEATURE = 64  # number of filters to use
        # apply convolution
        L = DN_CONV(N_FEATURE, kernel_size=kernel, strides=2, padding="same")(INPUT)
        # Apply leaky ReLU activation with a slope of 0.2
        L = LEAKY_RELU(0.2)(L)

        # Apply convolution MAX_LAYERS times
        for _ in range(1, MAX_LAYERS):
            N_FEATURE *= 2  # double the number of filters
            # Apply convolution
            L = DN_CONV(N_FEATURE, kernel_size=kernel, strides=2, padding="same")(L)
            # normalise
            L = BATNORM()(L, training=1)
            # Apply leaky ReLU activation with a slope of 0.2
            L = LEAKY_RELU(0.2)(L)

        N_FEATURE *= 2  # double the number of filters
        L = ZeroPadding2D(1)(L)  # pads the model with 0s with a thickness of 1
        # Apply convolution
        L = DN_CONV(N_FEATURE, kernel_size=kernel, padding="valid")(L)
        # normalise
        L = BATNORM()(L, training=1)
        # Apply leaky ReLU activation with a slope of 0.2
        L = LEAKY_RELU(0.2)(L)

        N_FEATURE = 1
        L = ZeroPadding2D(1)(L)  # pads the model with 0s with a thickness of 1
        # Apply sigmoid activation
        L = DN_CONV(
            N_FEATURE, kernel_size=kernel, padding="valid", activation="sigmoid"
        )(L)

    return Model(inputs=[INPUT_A, INPUT_B], outputs=L)


def BLOCK(X, S, NF_IN, USE_BATNORM=True, NF_OUT=None, NF_NEXT=None, kernel=4):
    MAX_N_FEATURE = 64 * 8  # max number of filters to use
    # Encoder: (decreasing size)
    assert S >= 2 and S % 2 == 0
    if NF_NEXT is None:  # number of filters in the next layer?
        # set number of filters to twice the number of filters in the
        # input, if it isn't more than the max number of filters
        NF_NEXT = min(NF_IN * 2, MAX_N_FEATURE)
    if NF_OUT is None:
        NF_OUT = NF_IN
    # Apply convolution
    X = DN_CONV(
        NF_NEXT,
        kernel_size=kernel,
        strides=2,
        # don't use a bias if batch normalisation will be done
        # later, or if s > 2
        use_bias=(not (USE_BATNORM and S > 2)),
        padding="same",
    )(X)
    if S > 2:
        # apply batch normalisation
        if USE_BATNORM:
            X = BATNORM()(X, training=1)
        # apply leaky ReLU with a slope of 0,2
        X2 = LEAKY_RELU(0.2)(X)
        # continue recursion until size = 2, halving size each time

        X2 = BLOCK(X2, S // 2, NF_NEXT)
        # combine X and X2
        # this gives the "skip connections" between the encoder layers
        # and decoder layers.

        X = Concatenate(axis=CH_AXIS)([X, X2])

    # Decoder: (Increasing size)
    # This happens only when the recursive encoder has reached its maximum
    # depth (size = 2)
    # Note the minimum layer size is actually s = 4, as encoding stops when
    # s = 2

    # Apply ReLU activation
    X = Activation("relu")(X)

    # Apply deconvolution
    X = UP_CONV(NF_OUT, kernel_size=4, strides=2, use_bias=not USE_BATNORM)(X)
    X = Cropping2D(1)(X)
    # Batch normalisation
    if USE_BATNORM:
        X = BATNORM()(X, training=1)
    # apply dropout
    # Randomly drops units which helps prevent overfitting
    if S <= 8:
        X = Dropout(0.5)(X, training=1)
    return X


# The generator (based on the U-Net architecture)
def UNET_G(ISIZE, NC_IN, NC_OUT, FIXED_INPUT_SIZE=True, kernel=4):
    S = ISIZE if FIXED_INPUT_SIZE else None  # size
    X = INPUT = Input(shape=(S, S, NC_IN))  # The input
    # Apply the U-Net convolution, deconvolution (see above function)
    X = BLOCK(X, ISIZE, NC_IN, False, NF_OUT=NC_OUT, NF_NEXT=64, kernel=kernel)
    # Apply tanh activation
    X = Activation("tanh")(X)

    return Model(inputs=INPUT, outputs=[X])


def get_mask(size):
    w = h = size
    center = (int(w / 2), int(h / 2))
    radius = w / 2 + 1
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


# The discriminator model
NET_D = BASIC_D(ISIZE, NC_IN, NC_OUT, MAX_LAYERS, kernel=4)
# The generator model
NET_G = UNET_G(ISIZE, NC_IN, NC_OUT, kernel=4)

# tensor placeholders?
REAL_A = NET_G.input  # generator input (AIA)
FAKE_B = NET_G.output  # generator output (fake HMI)
REAL_B = NET_D.inputs[1]  # descriminator input (real HMI)

# output of the discriminator for AIA and real HMI
OUTPUT_D_REAL = NET_D([REAL_A, REAL_B])
# output of the discriminator for AIA and fake HMI
OUTPUT_D_FAKE = NET_D([REAL_A, FAKE_B])

# set initial values for the loss
# ones_like creates a tensor of the same shape full of ones
# zeros_like creates a tensor of the same shape full of zeros
# as the discriminator gives the probability that the input is a real HMI
# picture, we want it to out put 1 when the input is real and 0 when the
# input is fake.
LOSS_D_REAL = LOSS_FN(OUTPUT_D_REAL, K.ones_like(OUTPUT_D_REAL))
LOSS_D_FAKE = LOSS_FN(OUTPUT_D_FAKE, K.zeros_like(OUTPUT_D_FAKE))
# while the generator, we want the discriminator to guess that the
# generator output is the real HMI, which corresponds to the discriminator
# outputting 1:
LOSS_G_FAKE = LOSS_FN(OUTPUT_D_FAKE, K.ones_like(OUTPUT_D_FAKE))

# total average difference between the real and generated HMIs
LOSS_L = K.mean(K.abs(FAKE_B - REAL_B))

# Total loss of the discriminator
LOSS_D = LOSS_D_REAL + LOSS_D_FAKE
# gives the updates for the discriminator training
TRAINING_UPDATES_D = Adam(lr=2e-4, beta_1=0.5).get_updates(
    LOSS_D, NET_D.trainable_weights
)
# creates a function that trains the discriminator
NET_D_TRAIN = K.function([REAL_A, REAL_B], [LOSS_D / 2.0], TRAINING_UPDATES_D)

# The total loss of G, which includes the difference between the real and
# generated HMIs, as well as the loss because of the descriminator
LOSS_G = LOSS_G_FAKE + 100 * LOSS_L

# operation to update the gradient of the generator using the adam optimizer
TRAINING_UPDATES_G = Adam(lr=2e-4, beta_1=0.5).get_updates(
    LOSS_G, NET_G.trainable_weights
)
# function to train the generator
NET_G_TRAIN = K.function([REAL_A, REAL_B], [LOSS_G_FAKE, LOSS_L], TRAINING_UPDATES_G)
