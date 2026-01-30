import gymnasium as gym
import numpy as np
import random
import time

env = gym.make("FrozenLake-v1", is_slippery=False)
q_table=np.zeros((env.observation_space.n, env.action_space.n))
learning_rate = 0.1
discount_factor = 0.99 
epsilon = 1.0
epsilon_decay=0.001
min_epsilon=0.01
total_episodes=1000

for episode in range(total_episodes):
    state,info = env.reset()
    done=False
    while not done:
        if random.uniform(0,1)>epsilon:
            action=env.action_space.sample() #Explore
        else:
            action=np.argmax(q_table[state]) #Exploit

        next_state,reward,terminated,truncated,info = env.step(action)
        done = terminated or truncated
        
        #Q(s,a)=Q(s,a)+alpha*[Reward+gamma*max(Q(s',a'))-Q(s,a)] => bellman eq
        best_next_q=np.max(q_table[next_state])
        q_table[state,action]=q_table[state,action]+learning_rate*(reward+discount_factor*best_next_q-q_table[state,action])
        
        state=next_state
        
    epsilon=max(min_epsilon,epsilon-epsilon_decay)
print("Training finished. Optimal Q-Table generated.")
env.close()

num_test_episode=5
for episode in range(num_test_episode):
    state,info=env.reset()
    done=False
    print(f"Episode {episode+1}:")
    while not done:
        action=np.argmax(q_table[state])
        state,reward,terminated,truncated,info = env.step(action)
        done = terminated or truncated
        time.sleep(0.5)
    
    if reward == 1:
        print("Success! Reached the Goal.")
    else:
        print("Failure. Fell into a hole.")
env.close()