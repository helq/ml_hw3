# Analysis #

The best model seems not to be the model proposed in @norb04, but the more simple
convolutional model adapted from other problem (MNIST). The difference in performance
between the model proposed and my implementantion shows how difficult is to reproduce
the results claimed in many papers on machine learning, but specially on Deep Learning,
where the implementantion details really matter.

It is iteresting how all testing accuracy scores are bigger than the validation accuracy
scores. This shouldn't happen, but because it always happens, I may conjecture that
the set used for validation was slightly more difficult to learn than the set used for
testing, even thought both come from the same dataset. A reason why this could have
happened, is because I didn't randomize the set given for testing, I just broke the set in
two equal parts, one for validation and one for testing. There is, probably, some order in
which the data was stored in the testing dataset, leaving the "easiest" datapoints
(images) at last.

<!-- vim:set filetype=markdown.pandoc : -->
