import numpy as np
import pickle

class simple_net(object):

    # Setup Class Variables
    def __init__(self, seed=1, verbose=False):
        self.verbose = verbose
        self.data_in = []
        self.desired_output = []
        self.layer_count = None
        self.random = np.random.seed(seed)
        self.synapse = []
        self.dataset_gain = 1
        self.dataset_bias = 0
        self.represented_as_ascii = False
        self.desired_ascii_output = None

    # Set Float Precision in Logging
    def do_logging_prettier(self, enable):
        if enable is True:
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # Guess what this does
    def log(self, a_string='Value', data=''):
        if self.verbose is True:
            print '\n{!s}: \n{!s}'.format(a_string, str(data))

    # Output the result and desired output for comparison
    def get_result(self, layer):
        store_verbose_setting = self.verbose
        self.verbose = True
        scaled_data_out = (self.dataset_gain*layer[self.layer_count-1]) - self.dataset_bias
        scaled_desired_out = (self.dataset_gain * self.desired_output) - self.dataset_bias
        error = scaled_desired_out - scaled_data_out
        self.log('Net Output', layer[self.layer_count-1])
        self.log('Desired Output', self.desired_output)
        self.log('Scaled Net Output', scaled_data_out)
        self.log('Scaled Desired Output', scaled_desired_out)
        self.log('Error', error)
        if np.nanmax(error) < 0.1 and np.nanmin(error) < 0.1:
            self.log('Max Error Less than 10%')
        if self.represented_as_ascii is True:
            output_string = []
            for row in scaled_data_out:
                a_string = ""
                for value in row:
                    a_string += (chr(int(value+0.1))) # HACK: Values occasionally approac limit .999
                output_string.append(a_string)
            self.log('Ascii Desired Output', self.desired_ascii_output)
            self.log('Ascii Net Result',output_string)
        self.verbose = store_verbose_setting


    # Sigmoid Function maps input to values between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of Sigmoid gives measure of confidence to be applied to calculated error during training
    def derivative_of_sigmoid(self, layer):
        return layer * (1 - layer)

    # Full Batch Prediction Function
    def forward_propagation(self, layer, iteration=0):
        self.log('Entering Forward Propagation Loop', iteration)
        layer[0] = self.data_in
        self.log('Layer {!s}'.format(0), layer[0])
        self.log('Synapse {!s}'.format(0), self.synapse[0])
        for j in range(0, self.layer_count-1):
            # Prediction using synapse values
            layer[j+1] = self.sigmoid(np.dot(layer[j], self.synapse[j]))
            self.log('Layer {!s}'.format(j+1), layer[j+1])
            self.log('Synapse {!s}'.format(j+1), self.synapse[j+1])
        return layer

    # Full Batch Update Backpropgation Function
    def backpropagation(self, layer, delta, error, iteration=0):
        self.log('Entering Backpropagation Loop', iteration)
        error[self.layer_count-1] = self.desired_output - layer[self.layer_count-1]
        delta[self.layer_count-1] = error[self.layer_count-1] * self.derivative_of_sigmoid(layer[self.layer_count-1])
        self.log('Error {!s}'.format(self.layer_count-1), error[self.layer_count-1])
        self.log('Delta {!s}'.format(self.layer_count-1), delta[self.layer_count-1])
        for j in reversed(range(0, self.layer_count-1)):
            error[j] = np.dot(delta[j+1], self.synapse[j].T)
            # Reduce the error of high confidence predictions
            delta[j] = error[j] * self.derivative_of_sigmoid(layer[j])
            self.log('Error {!s}'.format(j), error[j])
        return layer, delta, error

    # Update the Synapses based on the confidence of the value. Less confident => 'more updated'
    def update_synapses(self, layer, delta):
        self.log('Entering Update Synapse Loop')
        for j in range(0, self.layer_count-1):
            self.synapse[j] += np.dot(layer[j].T, delta[j+1])

    # Loop for Full Batch Backpropgation Training
    def train(self, layer_count, iterations, data_in=None, desired_output=None):
        self.log('Entering Training Loop')
        if data_in is not None or desired_output is not None:
            self.data_in = data_in
            self.desired_output = desired_output
        self.layer_count = layer_count
        self.synapse.append(2*np.random.random((self.data_in.shape[1], self.desired_output.shape[0])) - 1)
        for i in xrange(self.layer_count):
            self.synapse.append(2*np.random.random((self.desired_output.shape[0], self.desired_output.shape[1])) - 1)
            self.synapse.append(2*np.random.random((self.desired_output.shape[1], self.desired_output.shape[0])) - 1)
        layer = [None] * (self.layer_count)
        error = [None] * (self.layer_count)
        delta = [None] * (self.layer_count)
        for i in xrange(iterations):
            layer = self.forward_propagation(layer, i)
            layer, delta, error = self.backpropagation(layer, delta, error, i)
            self.update_synapses(layer, delta)
        self.get_result(layer)

    # Run Net, Requires a Synapse Array
    def run(self, data_in, layer_count, iterations):
        if self.synapse is None:
            self.log('Error', 'Please Load a Synapse')
            exit()
        self.layer_count = layer_count
        layer = [None] * (self.layer_count)
        layer[0] = data_in
        for i in xrange(iterations):
            layer = self.forward_propagation(layer, i)
        self.get_result(layer)
        return layer[self.layer_count-1]

    # Save Synapse for later use
    def save_synapse(self, file_name):
        self.log('Save Synapse', file_name)
        file_object = open(file_name, 'wb')
        pickle.dump(self.synapse, file_object)

    # Load a Synapse for reuse
    def load_synapse(self, file_name):
        self.log('Load Synapse', file_name)
        file_object = open(file_name, 'r')
        self.synapse = pickle.load(file_object)

    # Scale input and output datasets to (0,1) linearly using scaled_data = k * data - bias
    def scale_dataset_linear(self, data_in, desired_output):
        self.log('Linearly Scale Data')
        self.dataset_gain = np.ceil(max(np.nanmax(data_in), np.nanmax(desired_output)) - min(np.nanmin(data_in), np.nanmin(desired_output)))
        self.log('Dataset Gain', self.dataset_gain)
        self.dataset_bias = np.abs(min(np.nanmin(data_in), np.nanmin(desired_output)))
        self.log('Dataset Bias', self.dataset_bias)
        self.data_in = (data_in + self.dataset_bias) / self.dataset_gain
        self.desired_output = (desired_output + self.dataset_bias) / self.dataset_gain
        self.log('Original Data In', data_in)
        self.log('Scaled Data In', self.data_in)
        self.log('Original Desired Output', desired_output)
        self.log('Scaled Desired Output', self.desired_output)

    # Read in Ascii values and scale input / output for text
    # NB: currently requires all values in input strings to be same length
    def digest_ascii(self, data_in, desired_output):
        self.log('Digest Ascii Array', str(data_in) +'\n' + str(desired_output))
        self.represented_as_ascii = True
        self.desired_ascii_output = desired_output
        self.dataset_gain = 127
        #TODO: DRY this... derp
        for data in data_in:
            for cell in data:
                catcher = []
                for character in cell:
                    catcher.append(ord(character)/127.0) # won't accept nul
            self.data_in.append(np.asarray(catcher))

        for data in desired_output:
            for cell in data:
                catcher = []
                for character in cell:
                    catcher.append(ord(character)/127.0) # won't accept nul
                self.desired_output.append(np.asarray(catcher))
        self.data_in = np.asarray(self.data_in)
        self.desired_output = np.asarray(self.desired_output)
        self.log('Mapped Ascii Data In', self.data_in)
        self.log('Mapped Ascii Desired Output', self.desired_output)

# TODO: Create a sequential training method and rename 'train' to 'batch_train' or something similar
# TODO: Create a Map Scale version of linear dataset scale function
# TODO: third_generation.py:
# TODO: Gradient Descent
# TODO: Hinton's Dropout
