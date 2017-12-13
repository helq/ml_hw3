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
    with tf.variable_scope('FullyConnected', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        logits = tf.layers.dense(x, n_classes)

    return logits
