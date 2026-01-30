import gymnasium as gym 
import torch 
import torch.nn as nn
import torch.optim as optim


lr=1e-2 
gamma=0.99
episode=600

class PolicyNet(nn.Module):
    def __init__(self, obs_dim,act_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(obs_dim,128),
            nn.ReLU(),
            nn.Linear(128,act_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

    def act(self,state):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action=dist.sample()
        return action.item(), dist.log_prob(action)

env=gym.make("CartPole-v1")
obs_dim=env.observation_space.shape[0]
act_dim=env.action_space.n

policy=PolicyNet(obs_dim,act_dim)
optimizer=optim.Adam(policy.parameters(),lr=lr)


for ep in range(100):
    state,info=env.reset()
    log_probs,rewards=[],[]
    terminated=False
    truncated=False
    while not (terminated or truncated):
        state_t = torch.tensor(state,dtype=torch.float32)
        action,log_prob=policy.act(state_t)
        next_state,reward,terminated,truncated,info = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
        
    returns=[]
    G=0
    for r in reversed(rewards):
        G=r+gamma*G
        returns.insert(0,G)
    returns = torch.tensor(returns,dtype=torch.float32)
    returns = (returns - returns.mean())/(returns.std()+1e-8)

    loss=[]
    for log_prob,Gt in zip(log_probs,returns):
        loss.append(-log_prob*Gt)
    loss=torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode {ep}, total reward: {sum(rewards)}")
env.close()
