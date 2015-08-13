import numpy as np
from second_generation import second_generation

#########################################################
import os # not neccesary, just a useful delineator :-P #
# comment this if you have no sense of humour           #
os.system('echo Go, Neural Network, Go! | cowsay | lolcat')  #
#########################################################

data_in = np.array([[0,0,1,0],[0,1,1,0],[1,0,1,1],[1,1,1,1]])

desired_output = np.array([[0, 1, 1, 1],[1, 0, 1, 0],[1, 1, 1, 1],[0, 0, 0, 1]])

layer_count = 5

seed = 1

verbose = True

my_net = second_generation(seed, verbose)

my_net.do_logging_prettier(True)

iterations = 500

my_net.train(data_in, desired_output, layer_count, iterations)

my_net.save_synapse('test.synapse')

my_net.load_synapse('test.synapse')

my_net.run(data_in, layer_count, iterations)
