# Optimization/Training Algorithm #

All Neural Networks were implemented on Python using the Tensorflow library. All models
were trained using the Adam algorithm with parameters: learning rate of $0.001$, $\beta_1$
of $0.9$, $\beta_2$ of $0.999$, $\epsilon$ of $1\times10^{-8}$, and no locking. All
values, except for the learning rate, were the standard values setted by default by the
library.

Dropout was only used for one of the models (for the biggest model, based on a model that
has 98% accuracy on the MNIST dataset). The loss function used is the cross entropy
between the output of the networks and the hot vectors associated to the real vector value.

The batch size for training was setted to 128. This value was selected, because it was the
preferred in many examples of convolutional network training.

<!-- vim:set filetype=markdown.pandoc : -->
