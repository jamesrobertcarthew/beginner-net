# this code is inspired from http://iamtrask.github.io/2015/07/12/basic-python-network/
# Now I want to make more layers

import numpy as np

class first_generation(object):

    def __init__(self, data_in, desired_output, layer_count, seed, verbose):
        self.verbose = verbose
        self.data_in = data_in
        self.desired_output = desired_output
        self.layer_count = layer_count
        self.random = np.random.seed(seed)
        # initialize weights randomly with mean 0
        self.synapse = []
        for i in xrange(self.layer_count):
            self.synapse.append(2*np.random.random((self.data_in.shape[1], 1)) - 1)

    def sigmoid(self, x):
        self.log('x in sigmoid', x)
        return 1 / (1 + np.exp(-x))

    def derivative_of_sigmoid(self, layer):
        return layer * (1 - layer)

    def log(self, a_string, data=''):
        if self.verbose is True:
            print '\n{!s}: \n{!s}'.format(a_string, str(data))

    def train(self, iterations):
        layer = [None]*self.layer_count
        for i in xrange(iterations):
            layer[0] = self.data_in
            self.log('Layer Zero',layer[0])
            self.log('Synapse Zero', self.synapse[0])
            for j in range(0, self.layer_count-1):
                self.log("Layer {!s}".format(j), layer[j])
                self.log("Synapse {!s}".format(j), self.synapse[j])
                layer[j+1] = np.dot(layer[0], self.synapse[j])

            # a reverse for loop to weight the synapses and stuff

if __name__ == '__main__':
    # input dataset
    data_in = np.array([  [0,0,1],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1] ])

    # output dataset
    desired_output = np.array([[0,0,1,1]]).T

    # number of layers
    layer_count = 5

    # random seed
    seed = 3

    # toggle verbose
    verbose = True

    # create my_net
    my_net = first_generation(data_in, desired_output, layer_count, seed, verbose)

    # training iterations
    iterations = 2

    # train the net
    my_net.train(iterations)
