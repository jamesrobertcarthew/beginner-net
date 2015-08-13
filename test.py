import numpy as np
from second_generation import second_generation

#########################################################
import os # not neccesary, just a useful delineator :-P #
# comment this if you have no sense of humour           #
os.system('echo Go, Neural Network, Go! | cowsay | lolcat')  #
#########################################################

# input dataset
data_in = np.array([[0,0,1,0],[0,1,1,0],[1,0,1,1],[1,1,1,1]])

# output dataset
desired_output = np.array([[0, 1, 1, 1],[1, 0, 1, 0],[1, 1, 1, 1],[0, 0, 0, 1]])

# number of layers
layer_count = 5

# random seed
seed = 1

# toggle verbose
verbose = True

# create my_net
my_net = second_generation(seed, verbose)

# training iterations
iterations = 500

# train the net
my_net.train(data_in, desired_output, layer_count, iterations)

# save the synapse array to file
my_net.save_synapse('test.synapse')

# load synapse array
my_net.load_synapse('test.synapse')

# reapply synapse array
my_net.run(data_in, layer_count, iterations)
