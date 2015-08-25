# Simple Net
Adventures in Machine Learning!!!
## [First Generation](https://github.com/jamesrobertcarthew/machine-learning-experiments/tree/first-generation)
Converts [iamtrask](http://iamtrask.github.io/)'s 2 layer neural net into a class.
## [Second Generation](https://github.com/jamesrobertcarthew/machine-learning-experiments/tree/second-generation)
Provides a basic scalable class to create a backpropagation net with nothing fancy in terms of optimisation. Python Pickle lets me save and load the self.synapse values. Input, Desired Output and Layer count can be set (within reason).
#Overview:
##simple_net.py
* import numpy as np
* import pickle
* import os
* import inspect
## \_\_init\_\_(self, seed=1, verbose=False):
 Setup Class Variables
## set\_pretty\_log(self):
 Set Float Precision in Logging
## log(self, a\_string='Value', data=None):
## give\_result(self):
 Output the result and desired output for comparison
## get\_raw\_output(self):
 self.log('Desired Output', self.desired\_output)
## get\_scaled\_output(self):
 self.log('Scaled Desired Output', self.apply\_gain\_and\_bias(self.desired\_output))
## get\_ascii\_output(self):
 self.log('Ascii Desired Output', self.float\_to\_ascii(scaled\_desired\_out))
## sigmoid(self, x):
 Sigmoid Function maps input to values between 0 and 1
## derivative\_of\_sigmoid(self, layer):
 Derivative of Sigmoid gives measure of confidence to be applied to calculated error during training
## forward\_propagation(self, layer, iteration=0):
 Full Batch Prediction Function
 Prediction using synapse values
## backpropagation(self, layer, delta, error, iteration=0):
 Full Batch Update Backpropgation Function
 Reduce the error of high confidence predictions
## update\_synapses(self, layer, delta):
 Update the Synapses based on the confidence of the value. Less confident => 'more updated'
## initialise\_synapse(self):
 Initialise the Synapse structure
## train(self, layer\_count, iterations, data\_in=None, desired\_output=None):
 Loop for Full Batch Backpropgation Training
## run(self, layer\_count, iterations, data\_in=None):
 Run Neural Net, Requires a Synapse Array
## save\_synapse(self, file\_name):
 Save Synapse for later use
## load\_synapse(self, file\_name):
 Load a Synapse for reuse
## digest\_float(self, data\_in, desired\_output):
 Scale input and output datasets to (0,1) linearly using scaled\_data = k * data - bias
## apply\_gain\_and\_bias(self, data):
 Unscale the dataset after net
## digest\_ascii(self, data\_in, desired\_output=None):
 Read in Ascii values and scale input / output for text
 NB: currently requires all values in input strings to be same length
## ascii\_to\_float(self, ascii\_array):
 Convert an ascii array to a (normalised) float array
## float\_to\_ascii(self, data):
 Convert a normalised Float array to ascii characters
## TO DO:
* lets make this thing into more, smaller files and stuff... is getting out of control!!!
* Make the default mode 'run til convergence' with option to overtrain by setting iteration value (think about this and commit before hand cause you WILL fuck it up
* gradient descent and drop output
* CUDA dot product
* Seperate net from main loop via sockets

