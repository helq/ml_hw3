# Conclusions #

A simple convolutional model seems to work better than an elaborated model, but this could
change if a different algorithm for training is used or if the models are more fine tuned.

Any problem using images is a hard problem, and an annoying one, not only because the
problem is annoying but because using manipulating images is expensive and prone to error.

It is much more stressful to work with Neural Networks than with other methods/techniques
from Machine Learning, e.g., SVMs. It is very hard to know for sure how well a model works
on a problem, or how to modify a model for a similar problem into the current one. Not
knowing which direction to take or what went wrong when using NNs makes it very difficult
to think forward.

A lesson I learned was to randomize, always randomize, the datasets that we are given,
both training datasets and testing datasets. It is possible that in the order of the
elements of the dataset some information is hidden, information about how the data was
collected or stored. This is due mainly to prevent any weirdness in differences between
validation and testing accuracy scores, like the ones present in this report.

Another lesson I learned in this assignment was that: reproducing results from papers in
machine learning is hard!

<!-- vim:set filetype=markdown.pandoc : -->
