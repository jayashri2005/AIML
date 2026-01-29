import gymnasium as gym
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

env=gym.make("FrozenLake-v1", is_slippery=False)
Q=np.zeros((env.observation_space.n, env.action_space.n))

#observation-space discrete ->current position/state
#action-space discrete -> direction[up/down/left/right]
#model-free -> any env lack of complete knowledge
#model-based ->  had info about transition probabilities
#model = environment
#Qlearning -> incase of model-free,continuos observation;no knowledge about probability distribution
#sarsa -> incase of model-free
