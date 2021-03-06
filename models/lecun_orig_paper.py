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
    #dropout = params['num_classes']

    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x_ = x_dict['images']

        # NORB data input is a 1-D vector of 18432 features (2*96*96 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x_, shape=[-1, 96, 96, 2])

        # breaking binocular images into their monocular parts
        x_left, x_right = tf.split(x, [1,1], axis=3)

        # Convolution Layer with 32 filters and a kernel size of 5
        C1_map_monocular1 = tf.layers.conv2d(x_left,  2, 5, activation=activation)
        C1_map_monocular2 = tf.layers.conv2d(x_right, 2, 5, activation=activation)
        C1_map_binocular  = tf.layers.conv2d(x,       4, 5, activation=activation)
        # Max Pooling (down-sampling) with strides of 1 and kernel size of 4
        S2_map_monocular1 = tf.layers.max_pooling2d(C1_map_monocular1, 4, 1)
        S2_map_monocular2 = tf.layers.max_pooling2d(C1_map_monocular2, 4, 1)
        S2_map_binocular  = tf.layers.max_pooling2d(C1_map_binocular , 4, 1)

        C3_maps = [None for idx in range(24)]
        S2_mono1_splited = tf.split(S2_map_monocular1, 2, axis=3) # size 2
        S2_mono2_splited = tf.split(S2_map_monocular2, 2, axis=3) # size 2
        S2_bino_splited  = tf.split(S2_map_binocular,  4, axis=3) # size 4
        # Convolution Layer with 24 filters and a kernel size of 3
        for (i, j, l1, l2, idx) in indices_maps():
            S2_maps_to_C3_map = tf.concat(
                [S2_mono1_splited[i], S2_mono2_splited[j], S2_bino_splited[l1], S2_bino_splited[l2]]
                , axis=3)
            # creating feature map from 4 feature maps, 2 from binocular map and 2 from monocular maps
            C3_maps[idx] = tf.layers.conv2d(S2_maps_to_C3_map, 1, 3, activation=activation)

        # A total of 24 feature maps
        C3 = tf.concat(C3_maps, axis=3)
        # Max Pooling (down-sampling) with strides of 1 and kernel size of 3
        S4 = tf.layers.max_pooling2d(C3, 3, 1)

        # Creating 80 feature maps with kernels of size 6
        C5 = tf.layers.conv2d(S4, 80, 6, activation=activation)

        # Flatten the data to a 1-D vector for the fully connected layer
        FC6 = tf.contrib.layers.flatten(C5)

        # Fully connected layer. IT ISN'T DEFINED/USED in the paper
        #FC6 = tf.layers.dense(FC6, 1024)

        # Apply Dropout (if is_training is False, dropout is not applied)
        #FC6 = tf.layers.dropout(FC6, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(FC6, n_classes)

    return out

def indices_maps():
    for i in range(2):
        for j in range(2):
            k = 0
            for l1 in range(3):
                for l2 in range(l1+1, 4):
                    idx = i*12 + j*6 + k
                    k += 1
                    #print( i, j, l1, l2, idx, k )
                    yield i, j, l1, l2, idx
