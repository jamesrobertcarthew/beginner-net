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
        self.mode = 'RAW'  # ,SCALED, ASCII, ...

    # Set Float Precision in Logging
    def do_logging_prettier(self):
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # Guess what this does
    def log(self, a_string='Value', data=''):
        if self.verbose is True:
            print '\n{!s}: \n{!s}'.format(a_string, str(data))

    # Output the result and desired output for comparison
    def give_result(self, layer):
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
        self.log('Ascii Net Output', self.float_to_ascii(scaled_data_out))
        self.log('Ascii Desired Output', self.float_to_ascii(scaled_desired_out))
        self.verbose = store_verbose_setting

    def get_raw_output(self, layer):
        pass

    def get_scaled_output(self, layer):
        pass

    def get_ascii_output(self, layer):
        pass

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
        self.give_result(layer)

    # Run Net, Requires a Synapse Array
    def run(self, data_in, layer_count, iterations):
        store_verbose_setting = self.verbose
        if verbose is not None:
            self.verbose = verbose
        if self.synapse is None:
            self.log('Error', 'Please Load a Synapse')
            exit()
        self.layer_count = layer_count
        layer = [None] * (self.layer_count)
        layer[0] = data_in
        for i in xrange(iterations):
            layer = self.forward_propagation(layer, i)
        self.verbose = store_verbose_setting
        self.give_result(layer)
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
        self.mode = 'SCALED'
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
        self.log('Digest Ascii Array - Data In', str(data_in))
        self.log('Digest Ascii Array - Desired Output', str(desired_output))
        self.mode = 'ASCII'
        self.dataset_gain = 127
        self.data_in = self.ascii_to_float(data_in)
        self.desired_output = self.ascii_to_float(desired_output)
        self.log('Mapped Ascii Data In', self.data_in)
        self.log('Mapped Ascii Desired Output', self.desired_output)

    # Convert an ascii array to a (normalised) float array
    def ascii_to_float(self, ascii_array):
        float_representation = []
        for data in ascii_array:
            catcher = []
            for cell in data:
                for character in cell:
                    catcher.append((ord(character))/127.0)  # Cause Float
            float_representation.append(np.asarray(catcher))
        return np.asarray(float_representation)

    # Convert a normalised Float array to ascii characters
    def float_to_ascii(self, data):
        string_representation = []
        for row in data:
            a_string = ""
            for value in row:
                a_string += (chr(int(value + 0.1)))  # HACK: Values occasionally approac limit .999
            string_representation.append(a_string)
        return np.asarray(string_representation)