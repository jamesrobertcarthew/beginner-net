# this code is inspired from http://iamtrask.github.io/2015/07/12/basic-python-network/
# I want to learn this shiz
# This attempt will convert trask's 11 line machine learner into a class
import numpy as np

class first_generation(object):

    def __init__(self, data_in, desired_output, layer_count, seed, verbose):
        self.verbose = verbose
        self.data_in = data_in
        self.desired_output = desired_output
        self.layer_count = layer_count
        self.random = np.random.seed(seed)
        # initialize weights randomly with mean 0
        self.synapse = 2*np.random.random((self.data_in.shape[1], 1)) - 1

    def sigmoid(self, x):
        self.log('x in sigmoid', x)
        return 1 / (1 + np.exp(-x))

    def derivative_of_sigmoid(self, layer):
        return layer * (1 - layer)

    def log(self, a_string, data=''):
        if self.verbose is True:
            print '\n{!s}: \n{!s}'.format(a_string, str(data))

    def train(self, iterations):
        for iter in xrange(iterations):
            # forward propagation
            l0 = self.data_in
            l1 = self.sigmoid(np.dot(l0, self.synapse))

            # how much did we miss?
            l1_error = self.desired_output - l1
            self.log("Error", l1_error)

            # multiply how much we missed by the
            # slope of the sigmoid at the values in l1
            l1_delta = l1_error * self.derivative_of_sigmoid(l1)
            self.log("Delta", l1_delta)

            # update weights
            self.synapse += np.dot(l0.T, l1_delta)

        self.log('Net Output', l1)
        self.log('Desired Output', self.desired_output)

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
    iterations = 100

    # train the net
    my_net.train(iterations)
