"""
Copyright 2017 Elkin Cruz
Copyright Aymeric Damien

Based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
from .tensorflow_models.research.slim.nets import mobilenet_v1

# Create the neural network
def conv_net(x_dict, reuse, is_training, params):
    n_classes = params['num_classes']
    #dropout = params['num_classes']

    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNetMobile', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x_ = x_dict['images']

        # NORB data input is a 1-D vector of 18432 features (2*96*96 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x_, shape=[-1, 96, 96, 2])

        logits, _ = mobilenet_v1.mobilenet_v1(
                        x, num_classes=n_classes, is_training=is_training,
                        depth_multiplier=0.50, prediction_fn=None)

    return logits
