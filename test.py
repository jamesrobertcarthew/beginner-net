import numpy as np
from second_generation import second_generation

# input dataset
data_in = np.array([  [0,0,1], [0,1,1],[1,0,1],[1,1,1] ])

# output dataset
desired_output = np.array([[0,0,1,1]]).T

# number of layers
layer_count = 3

# random seed
seed = 1

# toggle verbose
verbose = True

# create my_net
my_net = second_generation(data_in, desired_output, layer_count, seed, verbose)

# training iterations
iterations = 2

# train the net
my_net.train(iterations)
