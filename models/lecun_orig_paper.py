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
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
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
        C1_map_monocular1 = tf.layers.conv2d(x_left,  2, 5, activation=tf.nn.relu)
        C1_map_monocular2 = tf.layers.conv2d(x_right, 2, 5, activation=tf.nn.relu)
        C1_map_binocular  = tf.layers.conv2d(x,       4, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 1 and kernel size of 4
        S2_map_monocular1 = tf.layers.max_pooling2d(C1_map_monocular1, 4, 1)
        S2_map_monocular2 = tf.layers.max_pooling2d(C1_map_monocular2, 4, 1)
        S2_map_binocular  = tf.layers.max_pooling2d(C1_map_binocular , 4, 1)

        C3_maps = [None for idx in range(24)]
        S2_mono1_splited = tf.split(S2_map_monocular1, 2, axis=3)
        S2_mono2_splited = tf.split(S2_map_monocular2, 2, axis=3)
        S2_bino_splited  = tf.split(S2_map_binocular,  4, axis=3)
        # Convolution Layer with 24 filters and a kernel size of 3
        for i in range(2):
            for j in range(2):
                k = 0
                for l1 in range(3):
                    for l2 in range(l1+1, 4):
                        idx = i*12 + j*6 + k
                        k += 1

                        S2_maps_to_C3_map = tf.concat(
                            [S2_mono1_splited[i], S2_mono2_splited[j], S2_bino_splited[l1], S2_bino_splited[l2]]
                            , axis=3)
                        # creating feature map from 4 feature maps, 2 from binocular map and 2 from monocular maps
                        C3_maps[idx] = tf.layers.conv2d(S2_maps_to_C3_map, 1, 3, activation=tf.nn.relu)

        # A total of 24 feature maps
        C3 = tf.concat(C3_maps, axis=3)
        # Max Pooling (down-sampling) with strides of 1 and kernel size of 3
        S4 = tf.layers.max_pooling2d(C3, 3, 1)

        # Creating 80 feature maps with kernels of size 6
        C5 = tf.layers.conv2d(S4, 80, 6, activation=tf.nn.relu)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(C5)

        # Fully connected layer. IT ISN'T DEFINED/USED in the paper
        #fc1 = tf.layers.dense(fc1, 1024)

        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode, params):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, params['num_classes'], params['dropout'], reuse=False, is_training=True)
    logits_test  = conv_net(features, params['num_classes'], params['dropout'], reuse=True, is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    #pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    else:
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs
