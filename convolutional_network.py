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

model_functions = {
    'mnist-modified': mnist_conv.model_fn
}

if __name__ == '__main__':
    from loaddataset import load_set

    model_name = 'mnist-modified'

    # Training Parameters
    num_steps = 2000
    batch_size = 128

    # Network Parameters
    #num_input = 96*96 # NORB image size

    config = tf.contrib.learn.RunConfig(
        save_checkpoints_steps=500
    )

    params = {
        'num_classes': 5, # NORB total classes
        'dropout': 0.75, # Dropout, probability to keep units
        'learning_rate': 0.001,
    }

    # Build the Estimator
    model_fn = model_functions[model_name]
    model_dir = 'models-results/{}'.format(model_name)
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir, config=config, params=params)

    train_imgs, train_labels = load_set('training')

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': train_imgs}, y=train_labels,
        batch_size=batch_size, num_epochs=None, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=num_steps)

    test_imgs, test_labels = load_set('testing')
    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': test_imgs}, y=test_labels,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)

    print("Testing Accuracy:", e['accuracy'])
