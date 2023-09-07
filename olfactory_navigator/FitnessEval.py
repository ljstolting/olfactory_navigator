#A short python function which evaluates the fitness 
#of a genome spit out by the C++ evolutionary algorithm
import numpy as np

def Olf_Nav(genome,start_pos,neuron_init=np.array([0,0,0])):
    return 0