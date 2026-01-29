import gymnasium as gym
import numpy as np 
import matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

matplotlib.use('Agg')
import matplotlib.pyplot as plt

env=gym.make("FrozenLake-v1", is_slippery=False)

# Create Neural Network instead of Q-table
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(16,)),  # Input: 16 states
    layers.Dense(16, activation='relu'),                    # Hidden layer
    layers.Dense(4, activation='linear')                     # Output: 4 actions
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Function to get Q-values from neural network
def get_q_values(state):
    state_onehot = np.zeros(16)
    state_onehot[state] = 1  # One-hot encoding
    return model.predict(state_onehot.reshape(1, -1), verbose=0)[0]

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
            action=np.argmax(get_q_values(state)) #explot using argmax
    
        next_state,reward,done,_,info=env.step(action) #move  agent into env 

        # Get current Q-values from neural network
        current_q = get_q_values(state)
        
        # Calculate target using Bellman equation
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(get_q_values(next_state))
        
        # Create target array (copy current Q-values, update only taken action)
        target_q = current_q.copy()
        target_q[action] = target
        
        # Train neural network
        state_onehot = np.zeros(16)
        state_onehot[state] = 1
        model.fit(state_onehot.reshape(1, -1), target_q.reshape(1, -1), 
                  verbose=0, epochs=1)
        
        # Calculate delta for convergence check
        delta_change = abs(current_q[action] - target)
        max_delta = max(max_delta, delta_change)
        
        state=next_state
        if done: #if reached -> true stop
            break
    epsilon=max(epsilon_min, epsilon*0.9)
    if max_delta<threshold and epsilon<0.05:
        print(f"Converged at episode {episode} with epsilon={epsilon:.4f}")
        break

print("Deep Q-Learning Training Completed!")

# Test the trained neural network
print("\nTesting trained Deep Q-Network:")
state,_=env.reset()
done=False
total_reward=0
steps=0

while not done and steps < 20:  # Prevent infinite loop
    action=np.argmax(get_q_values(state))
    state,reward,done,_,info=env.step(action)
    total_reward+=reward
    steps+=1

print(f"Total reward: {total_reward}")
print("Success!" if total_reward==1 else "Failed!")

"""
total reward (G)= r0+r1+r2+...
discounted reward (Gt)= r0+γ*r1++γ^2*r2+...
discounted reward (Gt)= r0+γ(Gt)
Q(s,a)=E(r0+γ*G1)

expected rewards/val = target

Deep Q-Learning Changes:
- Q-table → Neural Network
- Direct lookup → model.predict()
- Direct assignment → model.fit()
- Same Bellman equation logic
- Same exploration/exploitation
"""
