"""import torch 
from torch.distributions import Categorical
probs=torch.tensor([0.2,0.6,0.1,0.1])
dist=Categorical(probs)

ls=[]
for k in range(10):
    action=dist.sample()
    ls.append(action.item())

print(ls)

log_prob=dist.log_prob(action)
print(log_prob)
"""

import gymnasium as gym 
import torch 
import torch.nn as nn
import torch.optim as optim 
import numpy as np

env=gym.make("Hopper-v4")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print()
print(state_dim, action_dim)

env = gym.make("Hopper-v4", render_mode="human")

obs, info = env.reset()

for t in range(1000):
    action = env.action_space.sample()     # random action (later replace with your policy)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()