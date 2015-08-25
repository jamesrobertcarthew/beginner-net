import numpy as np
import pickle


class simple_net(object):

    def __init__(self, seed=1, verbose=False):
        # Setup Class Variables
        self.verbose = verbose
        self.data_in = []
        self.desired_output = []
        self.layer_count = None
        self.actual_raw_output = None
        self.random = np.random.seed(seed)
        self.synapse = []
        self.dataset_gain = 1
        self.dataset_bias = 0
        self.mode = 'RAW'  # ,SCALED, ASCII, ...

    def set_pretty_log(self):
        # Set Float Precision in Logging
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    def log(self, a_string='Value', data=None):
        if self.verbose is True:
            if data is None:
                print '\n{!s}\n'.format(a_string)
            else:
                print '\n{!s}: \n{!s}'.format(a_string, str(data))

    def give_result(self):
        # Output the result and desired output for comparison
        store_verbose_setting = self.verbose
        self.verbose = True
        error = self.desired_output - self.actual_raw_output
        data_mode = {
            'RAW': self.get_raw_output,
            'SCALED': self.get_scaled_output,
            'ASCII': self.get_ascii_output
        }
        func = data_mode.get(self.mode, lambda: "nothing")
        output = func()
        if np.nanmax(error) < 0.1 and np.nanmin(error) < 0.1:
            self.log('Max Error Less than 10%')
        else:
            self.log('Error Greater than 10%', error)
        self.verbose = store_verbose_setting
        return output

    def get_raw_output(self):
        self.log('Net Output', self.actual_raw_output)
        # self.log('Desired Output', self.desired_output)
        return self.actual_raw_output

    def get_scaled_output(self):
        self.log('Scaled Net Output', self.apply_gain_and_bias(self.actual_raw_output))
        # self.log('Scaled Desired Output', self.apply_gain_and_bias(self.desired_output))
        return self.apply_gain_and_bias(self.actual_raw_output)

    def get_ascii_output(self):
        scaled_data_out = self.apply_gain_and_bias(self.actual_raw_output)
        scaled_desired_out = self.apply_gain_and_bias(self.desired_output)
        self.log('Ascii Net Output', self.float_to_ascii(scaled_data_out))
        # self.log('Ascii Desired Output', self.float_to_ascii(scaled_desired_out))
        return self.float_to_ascii(scaled_data_out)

    def sigmoid(self, x):
        # Sigmoid Function maps input to values between 0 and 1
        return 1 / (1 + np.exp(-x))

    def derivative_of_sigmoid(self, layer):
        # Derivative of Sigmoid gives measure of confidence to be applied to calculated error during training
        return layer * (1 - layer)

    def forward_propagation(self, layer, iteration=0):
        # Full Batch Prediction Function
        layer[0] = self.data_in
        for j in range(0, self.layer_count-1):
            # Prediction using synapse values
            layer[j+1] = self.sigmoid(np.dot(layer[j], self.synapse[j]))
        return layer

    def backpropagation(self, layer, delta, error, iteration=0):
        # Full Batch Update Backpropgation Function
        error[self.layer_count-1] = self.desired_output - layer[self.layer_count-1]
        delta[self.layer_count-1] = error[self.layer_count-1] * self.derivative_of_sigmoid(layer[self.layer_count-1])
        for j in reversed(range(0, self.layer_count-1)):
            error[j] = np.dot(delta[j+1], self.synapse[j].T)
            # Reduce the error of high confidence predictions
            delta[j] = error[j] * self.derivative_of_sigmoid(layer[j])
        return layer, delta, error

    def update_synapses(self, layer, delta):
        # Update the Synapses based on the confidence of the value. Less confident => 'more updated'
        for j in range(0, self.layer_count-1):
            self.synapse[j] += np.dot(layer[j].T, delta[j+1])

    def initialise_synapse(self):
        # Initialise the Synapse structure
        self.synapse.append(2*np.random.random((self.data_in.shape[1], self.desired_output.shape[0])) - 1)
        for i in xrange(self.layer_count):
            self.synapse.append(2*np.random.random((self.desired_output.shape[0], self.desired_output.shape[1])) - 1)
            self.synapse.append(2*np.random.random((self.desired_output.shape[1], self.desired_output.shape[0])) - 1)

    def train(self, layer_count, iterations, data_in=None, desired_output=None):
        # Loop for Full Batch Backpropgation Training
        if data_in is not None or desired_output is not None:
            self.data_in = data_in
            self.desired_output = desired_output
        self.layer_count = layer_count
        if self.synapse == []:
            self.initialise_synapse()
        layer = [None] * (self.layer_count)
        error = [None] * (self.layer_count)
        delta = [None] * (self.layer_count)
        for i in xrange(iterations):
            layer = self.forward_propagation(layer, i)
            layer, delta, error = self.backpropagation(layer, delta, error, i)
            self.update_synapses(layer, delta)
        self.actual_raw_output = layer[self.layer_count-1]
        self.give_result()

    def run(self, layer_count, iterations, data_in=None):
        # Run Neural Net, Requires a Synapse Array
        if data_in is not None:
            self.data_in = data_in
        store_verbose_setting = self.verbose
        if self.synapse == []:
            exit()
        self.layer_count = layer_count
        layer = [None] * (self.layer_count)
        layer[0] = self.data_in
        for i in xrange(iterations):
            layer = self.forward_propagation(layer, i)
        self.actual_raw_output = layer[self.layer_count-1]
        self.give_result()
        return self.actual_raw_output

    def save_synapse(self, file_name):
        # Save Synapse for later use
        file_object = open(file_name, 'wb')
        pickle.dump(self.synapse, file_object)

    def load_synapse(self, file_name):
        # Load a Synapse for reuse
        file_object = open(file_name, 'r')
        self.synapse = pickle.load(file_object)

    def digest_float(self, data_in, desired_output):
        # Scale input and output datasets to (0,1) linearly using scaled_data = k * data - bias
        self.mode = 'SCALED'
        self.dataset_gain = np.ceil(max(np.nanmax(data_in), np.nanmax(desired_output)) - min(np.nanmin(data_in), np.nanmin(desired_output)))
        self.dataset_bias = np.abs(min(np.nanmin(data_in), np.nanmin(desired_output)))
        self.data_in = (data_in + self.dataset_bias) / self.dataset_gain
        self.desired_output = (desired_output + self.dataset_bias) / self.dataset_gain

    def apply_gain_and_bias(self, data):
        # Unscale the dataset after net
        return (self.dataset_gain * data) - self.dataset_bias

    def digest_ascii(self, data_in, desired_output=None):
        # Read in Ascii values and scale input / output for text
        # NB: currently requires all values in input strings to be same length
        self.mode = 'ASCII'
        self.dataset_gain = 127
        self.data_in = self.ascii_to_float(data_in)
        if desired_output is not None:
            self.desired_output = self.ascii_to_float(desired_output)

    def ascii_to_float(self, ascii_array):
        # Convert an ascii array to a (normalised) float array
        float_representation = []
        for data in ascii_array:
            catcher = []
            for string in data:
                for character in string:
                    catcher.append((ord(character))/127.0)
            float_representation.append(np.asarray(catcher))
        return np.asarray(float_representation)

    def float_to_ascii(self, data):
        # Convert a normalised Float array to ascii characters
        string_representation = []
        for row in data:
            a_string = ""
            for value in row:
                a_string += (chr(int(value + 0.1)))  # HACK: Values occasionally approac limit .999
            string_representation.append(a_string)
        return np.asarray(string_representation)

# TODO: lets make this thing into more, smaller files and stuff... is getting out of control!!!
# TODO: Make the default mode 'run til convergence' with option to overtrain by setting iteration value (think about this and commit before hand cause you WILL fuck it up
# TODO: gradient descent and drop output
# TODO: CUDA dot product
# TODO: Seperate net from main loop via sockets
