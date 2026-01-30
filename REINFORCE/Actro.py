from pyexpat import features
from altair import value
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
#import torch.nn.functional as F 

class ActroCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActroCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU()
        )

        self.actor=nn.Sequential(
            nn.Linear(128,output_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic=nn.Linear(128,1)
    
    def forward(self, state):
        features= self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value


env=gym.make("CartPole-v1")
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

print(state_dim,action_dim)

model = ActroCritic(state_dim, 128, action_dim)
actor_optimizer=optim.Adam(model.actor.parameters(),lr=0.001)
critic_optimizer=optim.Adam(model.critic.parameters(),lr=0.005)

env.reset()

episodes=1000
gamma=0.99

for episode in range(episodes):
    state, _ = env.reset()
    log_probs,values,rewards=[],[],[]
    done=False
    while not done:
        state_tensor = torch.FloatTensor(state)
        action_probs,state_value = model(state_tensor)
        action = torch.multinomial(action_probs,1).item()
        log_prob = torch.log(action_probs[action])
        log_probs.append(log_prob)
        values.append(state_value)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

    returns, advantage=[],[]
    G=0
    for i in reversed(range(len(rewards))):
        G=rewards[i]+(gamma*G if i<len(rewards)-1 else 0)
        returns.insert(0,G)

    returns = torch.tensor(returns)
    values = torch.cat(values)
    advantage=returns-values

    actor_loss = -torch.sum(torch.stack(log_probs) * advantage.detach())
    critic_loss = torch.nn.functional.mse_loss(values, returns)
    
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    total_loss = actor_loss + critic_loss
    total_loss.backward()
    

    actor_optimizer.step()
    critic_optimizer.step()

    if episode%100==0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")