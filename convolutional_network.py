"""
Copyright 2017 Elkin Cruz
Copyright Aymeric Damien

Based on: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

import models.mnist_modified as mnist_conv
import models.lecun_orig_paper as lecun_orig_conv
import models.mobilenet as mobilenet

def generate_model_fn(conv_net, is_validating=True):
    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode, params):
        nonlocal conv_net
        nonlocal is_validating
        # Build the neural network
        # Because Dropout have different behavior at training and prediction time, we
        # need to create 2 distinct computation graphs that still share the same weights.
        logits_train = conv_net(features, reuse=False, is_training=True, params=params)
        logits_test  = conv_net(features, reuse=True, is_training=False, params=params)

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

            accuracy_name = 'accuracy' if is_validating else 'accuracy_testing'
            # TF Estimators requires to return a EstimatorSpec, that specify
            # the different ops for training, evaluating, ...
            estim_specs = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_classes,
                loss=loss_op,
                train_op=train_op,
                eval_metric_ops={accuracy_name: acc_op})

            return estim_specs
    return model_fn

model_functions = {
    'mnist-modified':  mnist_conv.conv_net,
    'lecun-orig-conv': lecun_orig_conv.conv_net,
    'mobilenet': mobilenet.conv_net,
}

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    from loaddataset import load_set

    training = True
    validation = True
    model_params = {
        #'model_name': 'mnist-modified',
        #'model_name': 'lecun-orig-conv',
        'model_name': 'mobilenet',
        #'activation': 'relu',
        #'activation': 'tanh',
    }

    # Training Parameters
    train_times = 4
    num_steps = 500 # num_steps * train_times == total number of steps
    batch_size = 128
    save_checkpoints_steps=500
    log_step_count_steps=50
    keep_checkpoint_max=15

    # Network Parameters
    #num_input = 96*96 # NORB image size

    config = tf.contrib.learn.RunConfig(
        save_checkpoints_steps=save_checkpoints_steps
      , log_step_count_steps=log_step_count_steps
      , keep_checkpoint_max=keep_checkpoint_max
    )

    activations = {
        'tanh': tf.nn.tanh,
        'relu': tf.nn.relu
    }
    params = {
        'num_classes': 5, # NORB total classes
        'dropout': 0.75, # Dropout, probability to keep units
        'learning_rate': 0.001,
    }
    if 'activation' in model_params:
        params['activation'] = activations[ model_params['activation'] ]

    model_dir = 'models-results/{model_name}'.format(**model_params)
    if 'activation' in model_params:
        model_dir += '-{activation}'.format(**model_params)

    print("Model dir: {}".format(model_dir))

    # Build the Estimator
    model_fn = generate_model_fn( model_functions[model_params['model_name']] )
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config, params=params)

    if training and validation: # skip training if you're testing (not validating)
        train_imgs, train_labels = load_set('training')

        # Define the input function for training
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'images': train_imgs}, y=train_labels,
            batch_size=batch_size, num_epochs=None, shuffle=True)

    test_imgs_, test_labels_ = load_set('testing')
    if validation:
        valid_imgs   = test_imgs_  [:11700]
        valid_labels = test_labels_[:11700]
        del test_imgs_, test_labels_

        # Define the input function for evaluating
        input_fn_valid = tf.estimator.inputs.numpy_input_fn(
            x={'images': valid_imgs}, y=valid_labels,
            batch_size=100, shuffle=False)

        # Training the Model
        # This is necessary to trigger saving the accuracy measure
        for _ in range(train_times):
            if training:
                model.train(input_fn, steps=num_steps)

            # Evaluating model
            # Use the Estimator 'evaluate' method
            evaluation_valid = model.evaluate(input_fn_valid)

            print("Validation Accuracy: {}".format(evaluation_valid['accuracy']))

    else: # testing
        model_fn_testing = generate_model_fn( model_functions[model_params['model_name']], is_validating=False)
        model_testing = tf.estimator.Estimator(model_fn_testing, model_dir=model_dir, config=config, params=params)

        # This is left for the last selected model, the accuracy value used to select the model
        # doesn't usually reflect the real accuracy, it may be very well a fluke
        test_imgs   = test_imgs_  [11700:]
        test_labels = test_labels_[11700:]
        del test_imgs_, test_labels_
        # Define the input function for evaluating
        input_fn_test = tf.estimator.inputs.numpy_input_fn(
            x={'images': test_imgs}, y=test_labels,
            batch_size=100, shuffle=False)
        # Use the Estimator 'evaluate' method
        evaluation_test = model_testing.evaluate(input_fn_test)
        print("Testing Accuracy: {}".format(evaluation_test['accuracy_testing']))
