## Training of some Neural Networks Architectures with the small NORB dataset ##

This repo was the result of several days of work to try to solve a problem for a class. If
you want to know what is this about, please read the pdf document that can be found under
the folder `paper`.

The code is has been left, unintentionally, undocumented. Let me ask you for forgiveness,
the code will stay undocumented, except for the few lines in this README.

## How to run and check the different models implemented and talked about in the report ##

First check that you have everything installed and runs smoothly, for this just execute:

~~~
python convolutional_network.py
~~~

If it runs for several minutes without any problem, and it creates a folder called
`models-results`.

If it doesn't run, check first that you have downloaded the dataset and have saved it in a
new folder called `dataset`. You can find the dataset in
<https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/>, download all `mat.gz` files.

If it still doesn't run, make sure that you have installed the following (python) libraries:

- Tensorflow
- Numpy

### Changing the model to test ###

There are a total of 5 different architectures to select from, to select an architecture
uncomment it at the lines 77-83 in the main file `convolutional_network.py`.

In lines 74 and 75 you can change the behaivour of the script, for example, you can skip
training to validate an already trained model.

The code shouldn't be too hard to read, I hope it is useful for somebody. Happy coding :)
