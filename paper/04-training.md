# Training and Analysis #

The results of training, validation and testing for each model can be seen next.

## One-layer Perceptron ##

This simple model (basically 5 unbalanced linear regression problems, each one classifying
one of the classes) is able to capture some information about the images, its testing
accuracy was 61.9%, which is significantly better than 20% if the the classification was
random.

In Figure \ref{perceptron_validation_accuracy}, the validation accuracy is shown.  The
perceptron goes from not being able to distinguish at step 50 (each step is an execution
of Adam in a batch of size 128) to (apparently) an accuracy measure of aprox. 60%-70%
after 1000 iterations.

Figure \ref{perceptron_loss} shows the change of the loss function as time goes. The loss
on training is smaller than in validation, as expected, but it never reaches zero, which
means that the model cannot memorize the training data.

![Validation Accuracy for One-layer Perceptron\label{perceptron_validation_accuracy}](imgs/perceptron_validation_accuracy.pdf)

![Loss value function for One-layer Perceptron\label{perceptron_loss}](imgs/perceptron_loss.pdf)

## Two-layer Perceptron ##

This model is a little more complex than the one-layer percetron, therefore it is able to
memomorize a little better as it can be seen in Figure \ref{onehidden_loss}. Its testing
accuracy was 70.4%, better indeed than one-layer (linear regression). The best model would
probably be the model at step 2100, because it seems that at around 2100 the loss is
smallest and more consistent than in 3000.

![Validation Accuracy for Two-layer Perceptron\label{onehidden_validation_accuracy}](imgs/onehidden_validation_accuracy.pdf)

![Loss value function for Two-layer Perceptron\label{onehidden_loss}](imgs/onehidden_loss.pdf)

## MNIST-Convolutional adaptation ##

There are two variations of this model, one using tanh as activation function and other
using relu. As it can be seen in Figure \ref{mnist_validation_accuracy}, the model with
activation tanh has a little problem when it is left too long optimizing. The testing
accuracy for the model tanh was 86.9% while the model relu was 90.4%.

In Figure \ref{mnist_relu_loss}, we can see that we arrive at a model that has memorized
the training data, and as time goes on, it seems that the model is overfitting (the loss
function increases for the validation set), probably a good model has been already found
with 500 steps.

![Validation Accuracy for MNIST adapted\label{mnist_validation_accuracy}](imgs/mnist_validation_accuracy.pdf)

![Loss value function for MNIST adapted (with relu activation)\label{mnist_relu_loss}](imgs/mnist_relu_loss.pdf)

## LeCun's model ##

This model was proposed by @norb04 when they introduced the NORB dataset. I hoped to
reproduce the same results they got, namely, around 7% error score (about 93% accuracy),
but the final result was a little disappointing. I got 90.1% accuracy on the testing set.
The validation accuracy on training can be seen in Figure \ref{lecun_validation_accuracy}.

The model I trained probably doesn't performs as well as it does in the original paper
because, probably, I'm using different training and initialization algorithms, and other
little details on implementations could be different from the paper.

![Validation Accuracy for LeCun's model\label{lecun_validation_accuracy}](imgs/lecun_validation_accuracy.pdf)

<!-- vim:set filetype=markdown.pandoc : -->
