# Models #

A total of 6 models were tested to try solve this problem. The first two model can be
considered as baseline, given their simplicity. The others involve modifications to two
convolutional networks.[^failures]

All models have the same inputs and outputs. The inputs are the $96 \times 96$ (times two)
pixels on the binocular images. The outputs are six real numbers, in inference the
output with the biggest value tells the class.

[^failures]: A small note about the current state of libraries for Deep Learning. I tried
  using more complicated pre-implemented models such as Mobilenet [@mobilenet17]. I
  thought that given these implementations are well tested, it should be better to use
  them than to implemented from scratch, because I could add many bugs to them. To my
  surprise, this proved to be harder than implementing a model from scratch. Tensorflow is
  a big library and it is in constant development and improvement, it is also a little
  "low level", and constantly new libraries/wrappers appear to make some tasks easier
  (e.g., creating layers for NNs, or log training process). This has incentivated the
  creation of multiple wrappers around Tensorflow, each one with its own idioms and
  ways to do their stuff. I failed completely trying to make use of one of such wrappers
  and couldn't try more complex models therefore :( .

## One-layer Perceptron ##

The first model can be seen as a generalization of linear regression, where 5 linear
regressions are performed simultaneously each one representing the likelihood that one
image belongs to a class or another. The activation function for this perceptron is
lineal.

## Two-layer Perceptron ##

The second model is a perceptron with one hidden layer. The hidden layer has 900 neurons
with activations tanh.

## MNIST-Convolutional adaptation ##

This model is an adaptation from the example code for convolutional networks in Tensorflow
Examples[^tf-examples]. The original code is teted on on the MNIST dataset and it gets a very high
testing accuracy, aproximately 98% accuracy.

[^tf-examples]: <https://github.com/aymericdamien/TensorFlow-Examples/blob/d43c58c948cb4fec189b13a39a620e54879a3495/examples/3_NeuralNetworks/convolutional_network.py>

The Network is defined by five layers C1, S2, C3, S4 and FC5. C1 is a convolutional
layer with input a binocular image ($96 \times 96$ twice), 32 filters and $5 \times 5$
kernels. S2 is a Max Pooling layer with kernel of 2 and strides of 2. C3 is a
convolutional layer with 64 filters and a kernel size of 3. S4 is a max pooling layer with
kernel 2 and strides of 2. FC5 is a fully connected layer (percetron) that takes the $24
\times 24 \times 64$ outputs from S4. FC5 outputs a value for each one of the 5 classes.

This model was tested with two different activation functions, tanh and relu.

## LeCun's model ##

This model is more complicated than MNIST-adaptation (above), but it is smaller on the
number of weights. This is the model defined in the original paper NORB was introduced
@norb04, they argue (in the paper) that this model can get up to 6% error testing.

The Network is defined by six layers: C1, S2, C3, S4, C5, and FC6. C1 is a layer composed
of 3 convolutional layers with kernels of size 5, the first two layers take a monocular
image (left or right) and output 2 filters, and the third layer takes the binocular
image and outputs 4 filters. S2 is a max pooling layer with kernel of size 4 and strides
of 1. C3 is a layer composed of 24 distinct convolutional layers with kernels of size 3,
each convolutional layer takes 4 filters ("images") from S2, one from the first monocular
filter, other from the second monocular filter, and two more from the 4 binocular filters.
S4 is a max pooling layer with kernel of size 3 and strides of 1. C5 is a convolutional
layer that takes all 24 filters and outputs 80 filters in total, with kernels of size 6.
FC6 is a fully connected layer that takes all outputs from the 80 filters and outputs 5
values for each one of the classes.

This model was tested with two different activation functions, tanh and relu.

<!-- vim:set filetype=markdown.pandoc : -->
