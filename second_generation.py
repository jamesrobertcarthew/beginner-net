# this code is inspired from http://iamtrask.github.io/2015/07/12/basic-python-network/

# HR manager in an infinite monkey cage

import numpy as np
import pickle

class second_generation(object):

    def __init__(self, seed=1, verbose=False):
        self.verbose = verbose
        self.data_in = None
        self.desired_output = None
        self.layer_count = None
        self.random = np.random.seed(seed)
        self.synapse = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_of_sigmoid(self, layer):
        return layer * (1 - layer)

    def log(self, a_string='Value', data=''):
        if self.verbose is True:
            print '\n{!s}: \n{!s}'.format(a_string, str(data))

    def forward_propagation(self, layer, iteration):
        self.log('Entering Forward Propagation Loop', iteration)
        layer[0] = self.data_in
        self.log('Layer {!s}'.format(0), layer[0])
        self.log('Synapse {!s}'.format(0), self.synapse[0])
        for j in range(0, self.layer_count-1):
            layer[j+1] = self.sigmoid(np.dot(layer[j], self.synapse[j]))
            self.log('Layer {!s}'.format(j+1), layer[j+1])
            self.log('Synapse {!s}'.format(j+1), self.synapse[j+1])
        return layer

    def backpropagation(self, layer, delta, error, iteration):
        self.log('Entering Backpropagation Loop', iteration)
        error[self.layer_count-1] = self.desired_output - layer[self.layer_count-1]
        delta[self.layer_count-1] = error[self.layer_count-1] * self.derivative_of_sigmoid(layer[self.layer_count-1])
        self.log('Error {!s}'.format(self.layer_count-1), error[self.layer_count-1])
        self.log('Delta {!s}'.format(self.layer_count-1), delta[self.layer_count-1])
        for j in reversed(range(0, self.layer_count-1)):
            error[j] = np.dot(delta[j+1], self.synapse[j].T)
            delta[j] = error[j] * self.derivative_of_sigmoid(layer[j])
            self.log('Error {!s}'.format(j), error[j])
        return layer, delta, error

    def update_synapses(self, layer, delta):
        self.log('Entering Update Synapse Loop')
        for j in range(0, self.layer_count-1):
            self.synapse[j] += np.dot(layer[j].T, delta[j+1])

    def train(self, data_in, desired_output, layer_count, iterations):
        self.log('Entering Training Loop')
        self.data_in = data_in
        self.desired_output = desired_output
        self.layer_count = layer_count
        for i in xrange(self.layer_count):
            self.synapse.append(2*np.random.random((self.data_in.shape[1], self.desired_output.shape[0])) - 1)
            self.synapse.append(2*np.random.random((self.data_in.shape[0], self.desired_output.shape[1])) - 1)
        layer = [None] * (self.layer_count)
        error = [None] * (self.layer_count)
        delta = [None] * (self.layer_count)
        for i in xrange(iterations):
            layer = self.forward_propagation(layer, i)
            layer, delta, error = self.backpropagation(layer, delta, error, i)
            self.update_synapses(layer, delta)
        self.show_result(layer)

    def show_result(self, layer):
        if self.verbose is True:
            self.log('Net Output', layer[self.layer_count-1])
            rounded_array = np.around(layer[self.layer_count-1])
            self.log('Rounded Output', rounded_array)
            self.log('Desired Output', self.desired_output)
            if np.array_equal(rounded_array, self.desired_output) is True:
                self.log('It worked!',':-)')
            else:
                self.log('Sorry, try again',':-(')

    def run(self, data_in, layer_count, iterations):
        self.layer_count = layer_count
        layer = [None] * (self.layer_count)
        layer[0] = data_in
        for i in xrange(iterations):
            layer = self.forward_propagation(layer, i)
        self.show_result(layer)
        return layer[self.layer_count-1]


    def save_synapse(self, file_name):
        self.log('Save Synapse', file_name)
        file_object = open(file_name, 'wb')
        pickle.dump(self.synapse, file_object)

    def load_synapse(self, file_name):
        self.log('Load Synapse', file_name)
        file_object = open(file_name, 'r')
        self.synapse = pickle.load(file_object)
