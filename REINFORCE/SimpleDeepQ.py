import gymnasium as gym
import numpy as np
import random

# Simple neural network using only numpy
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # Forward pass
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def train(self, x, target, learning_rate=0.01):
        # Simple gradient descent
        output = self.forward(x)
        
        # Calculate error
        error = output - target
        
        # Backpropagation (simplified)
        dW2 = np.dot(self.a1.T, error)
        db2 = np.sum(error, axis=0, keepdims=True)
        
        d_hidden = np.dot(error, self.W2.T)
        d_hidden[self.z1 <= 0] = 0  # ReLU derivative
        
        dW1 = np.dot(x.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Create neural network
nn = SimpleNeuralNetwork(16, 16, 4)

def get_q_values(state):
    state_onehot = np.zeros((1, 16))
    state_onehot[0, state] = 1
    return nn.forward(state_onehot)[0]

# Training parameters
learning_rate = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
num_episodes = 100

print("Starting Simple Deep Q-Learning...")

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(get_q_values(state))
        
        # Take action
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        # Get current Q-values
        current_q = get_q_values(state)
        
        # Calculate target
        if done:
            target = reward
        else:
            target = reward + gamma * np.max(get_q_values(next_state))
        
        # Create target array
        target_q = current_q.copy()
        target_q[action] = target
        
        # Train neural network
        state_onehot = np.zeros((1, 16))
        state_onehot[0, state] = 1
        target_array = target_q.reshape(1, 4)
        
        nn.train(state_onehot, target_array, learning_rate)
        
        state = next_state
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * 0.99)
    
    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

print("\nTraining completed!")

# Test the trained network
print("\nTesting trained network:")
state, _ = env.reset()
done = False
total_reward = 0
steps = 0

while not done and steps < 20:
    action = np.argmax(get_q_values(state))
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
    steps += 1

print(f"Total reward: {total_reward}")
print("Success!" if total_reward == 1 else "Failed!")
