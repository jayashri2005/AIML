import gymnasium as gym
import numpy as np 
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

env=gym.make("FrozenLake-v1", is_slippery=False)
Q=np.zeros((env.observation_space.n, env.action_space.n)) #Qtable [16,4]

"""
#observation-space discrete ->current position/state
#action-space discrete -> direction[up/down/left/right]
#model-free -> any env lack of complete knowledge
#model-based ->  had info about transition probabilities
#model = environment
#Qlearning -> incase of model-free,continuos observation;no knowledge about probability distribution
#sarsa -> incase of model-free
#qvalue la prob illa develop in dark -simply a guess wrk nd futer expectation ;
#  but value dev in iter la prob iruthuchu athuvum include pani val develop panom
#exploration -> explore out
#exploit -> find new place nd continuosly go there;keeping going to same thing
#lack of knowledge with env -> so we use explore&exploit

"""

#Hyperparameter
alpha=0.1
gamma=0.99
epsilon = 1
epsilon_max=1.0
epsilon_min=0.01
epsilon_decay=0.995
decay_rate=0.001
num_episodes=100
threshold=1e-3

 # Reset environment first!
for episode in range(num_episodes):
    state,_=env.reset()  # Unpack the tuple
    max_delta=0 
    while True:
        if np.random.rand()<epsilon:
            action=env.action_space.sample() #explore
        else:
            action=np.argmax(Q[state]) #explot using argmax
    
        next_state,reward,done,_,info=env.step(action) #move  agent into env 

        old_value=Q[state,action]
        #bellman eq V(s) = max_a [R(s,a) + γ * V(s')] 
        # instead of prob we have expectation [not fixed] -> so every nxt step target expect moving
        Q[state,action]=Q[state,action]+alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action]) #reward+...) -> target
        #like y-yp;alpha-learning rate;gamma-discount factor;Q[next_state] -> reward in Q val
        #use err to take feedback
        max_delta=max(max_delta,abs(old_value-Q[state,action]))
        state=next_state
        if done: #if reached -> true stop
            break
    epsilon=max(epsilon_min, epsilon*0.9)
    if max_delta<threshold and epsilon<0.05:
        print(f"Converged at episode {episode} with epsilon={epsilon:.4f}")
        break

"""
total reward (G)= r0+r1+r2+...
discounted reward (Gt)= r0+γ*r1++γ^2*r2+...
discounted reward (Gt)= r0+γ(Gt)
Q(s,a)=E(r0+γ*G1)

expected rewards/val = target

"""


