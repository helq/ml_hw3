"""
Copyright 2017 Elkin Cruz
Copyright Aymeric Damien

Based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Create the neural network
def conv_net(x_dict, reuse, is_training, params):
    activation = params['activation']
    n_classes = params['num_classes']
    dropout = params['dropout']

    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # NORB data input is a 1-D vector of 18432 features (2*96*96 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 96, 96, 2])

        # Convolution Layer with 32 filters and a kernel size of 5
        C1 = tf.layers.conv2d(x, 32, 5, activation=activation)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        S2 = tf.layers.max_pooling2d(C1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        C3 = tf.layers.conv2d(S2, 64, 3, activation=activation)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        S4 = tf.layers.max_pooling2d(C3, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        S4_flattened = tf.contrib.layers.flatten(S4)

        # Fully connected layer (in tf contrib folder for now)
        FC5 = tf.layers.dense(S4_flattened, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        FC5_dropout = tf.layers.dropout(FC5, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(FC5_dropout, n_classes)

    return out
