# this code is inspired from http://iamtrask.github.io/2015/07/12/basic-python-network/
# Now I want to make more layers

import numpy as np

class second_generation(object):

    def __init__(self, data_in, desired_output, layer_count, seed, verbose=False):
        self.verbose = verbose
        self.data_in = data_in
        self.desired_output = desired_output
        self.layer_count = layer_count
        self.random = np.random.seed(seed)
        self.synapse = []
        for i in xrange(self.layer_count):
            self.synapse.append(2*np.random.random((self.data_in.shape[1], 1)) - 1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_of_sigmoid(self, layer):
        return layer * (1 - layer)

    def log(self, a_string='Value', data=''):
        if self.verbose is True:
            print '\n{!s}: \n{!s}'.format(a_string, str(data))

    def forward_propagation(self, layer):
        layer[0] = self.data_in
        self.log('Layer 0',layer[0])
        self.log('Synapse 0', self.synapse[0])
        for j in range(0, self.layer_count-1):
            self.log("Layer {!s}".format(j), layer[j])
            self.log("Synapse {!s}".format(j), self.synapse[j])
            layer[j+1] = np.dot(layer[0], self.synapse[j])
        return layer

    def train(self, iterations):
        layer = [None]*self.layer_count
        error = [None]*self.layer_count
        delta = [None]*self.layer_count
        for i in xrange(iterations):
            layer = self.forward_propagation(layer)
            self.log('Enter Backpropagation Loop')
            error[self.layer_count-1] = self.desired_output - layer[self.layer_count-1]
            delta[self.layer_count-1] = error[self.layer_count-1] * self.derivative_of_sigmoid(layer[self.layer_count-1])
            self.log('Error {!s}'.format(self.layer_count-1), error[self.layer_count-1])
            self.log('Confidence Weighted Error {!s}'.format(self.layer_count-1), delta[self.layer_count-1])
            for j in reversed(range(0, self.layer_count-1)):
                self.log('Error {!s}'.format(j), error[j])
                self.log('Confidence Weighted Error {!s}'.format(j), delta[j])
