<!--\listoftodos-->

# Introduction and Dataset #

<!--
   -\inlinetodo{Explain what is this paper about, the problems presented in the homework, and
   -the data}
   -->

This document documents the stuff done to try to solve a problem of artificial vision (or
machine learning on vision).

The dataset used was NORB-small[^norburl]. The problem posed by NORB-small consists in
identify an object in an image from one of five categories. All images have a size of $96
\times 96$ pixels and have two channels, the images are binocular images, i.e., two
cameras, side by side, were used for a single shot. The five categories of objects are:
four-legged animals, human figures, airplanes, trucks, and cars.

[^norburl]: NORB-small can be found at <http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/>

The NORB-small dataset has 23400 images for training and 23400 images for testing. I
broke the testing set into two parts, each of the same size. The first 17700 images were
used in validation and the last 17700 were used in testing. Remember that for a binary
classification problem we can use the Chernoff equation to estimate the margin of the
testing error from the "real" error (which we could get if we had infinite datapoints).
With 17700 datapoints and a confidence of 95%, we know that the margin would be
aproximately $1.4\%$. Unfortunatelly the same calculations get very tricky for the case of
multiple classes, and therefore I could not give any certainty on the results obtained
here.

To solve the problem, I decided to use Neural Networks. A simple Neural Network
(Multilayer-Perceptron with one hidden layer) was used as "baseline", its accuracy was
$70.4\%$. Other models, such as Convolutional Networks were also tested.

The source code used to run the experiments can be found at <https://github.com/helq/ml_hw3>.

<!-- vim:set filetype=markdown.pandoc : -->
