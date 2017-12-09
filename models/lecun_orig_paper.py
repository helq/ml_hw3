"""
Copyright 2017 Elkin Cruz
Copyright Aymeric Damien

Based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x_dict, dropout, reuse, params):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']
        # NORB data input is a 1-D vector of 18492 features (2*96*96 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 96, 96, 2])

        # Store layers weight & bias
        weights = {
            # 5x5 conv, 2 inputs, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 2, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([24*24*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, params['num_classes']]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([params['num_classes']]))
        }

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def model_fn(features, labels, mode, params):
    dropout = params['dropout']

    logits_train = conv_net(features, tf.constant(dropout), reuse=False, params=params)
    logits_test  = conv_net(features, tf.constant(1.0),     reuse=True,  params=params)

    pred_classes = tf.argmax(logits_test, axis=1)

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

#if __name__ == '__main__':
#    # Training Parameters
#    learning_rate = 0.001
#    num_steps = 200
#    batch_size = 128
#    display_step = 10
#
#    # Network Parameters
#    num_input = 784 # MNIST data input (img shape: 28*28)
#    num_classes = 10 # MNIST total classes (0-9 digits)
#    #dropout = 0.75 # Dropout, probability to keep units
#
#    X, Y, train_op, loss_op, accuracy = build_model(0.8)
#
#    # Initialize the variables (i.e. assign their default value)
#    init = tf.global_variables_initializer()
#
#    # Start training
#    with tf.Session() as sess:
#
#        # Run the initializer
#        sess.run(init)
#
#        for step in range(1, num_steps+1):
#            batch_x, batch_y = mnist.train.next_batch(batch_size)
#            # Run optimization op (backprop)
#            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
#            if step % display_step == 0 or step == 1:
#                # Calculate batch loss and accuracy
#                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
#                                                                     Y: batch_y,
#                                                                     keep_prob: 1.0})
#                print("Step " + str(step) + ", Minibatch Loss= " + \
#                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                      "{:.3f}".format(acc))
#
#        print("Optimization Finished!")
#
#        # Calculate accuracy for 256 MNIST test images
#        print("Testing Accuracy:", \
#            sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
#                                          Y: mnist.test.labels[:256],
#                                          keep_prob: 1.0}))
