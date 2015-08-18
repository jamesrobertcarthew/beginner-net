import numpy as np
from simple_net import simple_net

#########################################################
import os # not neccesary, just a useful delineator :-P #
# comment this if you have no sense of humour           #
os.system('echo Go, Neural Network, Go! | cowsay | lolcat')  #
#########################################################

data_in = np.array([[0,0,-2,0],[0,1,1,0],[1,0,8,1],[1,1,1,1]])

desired_output = np.array([[0, 1, 1, 1],[1, 0, 1, 0],[1, -1, 5, 1],[0, 0, 0, 1]])

chars_in = np.array([['a bag bites a cat'], ['a hat bites a bat'], ['a cat bites a hat']])

desired_chars_out = np.array([['a cat is not a bat'], ['a dog is not a hat'], ['a rat is not a cat']])

layer_count = 5

seed = 1

verbose = False

my_net = simple_net(seed, verbose)

my_net.do_logging_prettier(True)

my_net.digest_ascii(chars_in, desired_chars_out)

# my_net.scale_dataset_linear(data_in, desired_output)

iterations = 500000

my_net.train(layer_count, iterations)

my_net.save_synapse('atest.synapse')

my_net.load_synapse('atest.synapse')

# my_net.run(data_in, layer_count, iterations)
