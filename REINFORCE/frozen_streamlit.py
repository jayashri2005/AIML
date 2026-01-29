import streamlit as st
from turtle import right
import gymnasium as gym 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np 

st.title("FrozenLake Environment Visualization")

print(gym.envs.registry.keys())
env=gym.make("FrozenLake-v1",is_slippery=False, render_mode="rgb_array")
env.reset()
rgb_array = env.render()

st.image(rgb_array, caption="FrozenLake Environment")

st.write(f"Observation Space: {env.observation_space.n}")
st.write(f"Action Space: {env.action_space.n}")

st.subheader("Environment Steps")
step_results = []
step_images = []

# Initial state
rgb_array = env.render()
step_images.append(rgb_array)
st.image(rgb_array, caption="Initial State")

step_results.append(env.step(2)) #right -2 
rgb_array = env.render()
step_images.append(rgb_array)
st.image(rgb_array, caption="Step 1: Right")

step_results.append(env.step(2)) #right
rgb_array = env.render()
step_images.append(rgb_array)
st.image(rgb_array, caption="Step 2: Right")

step_results.append(env.step(1)) #down
rgb_array = env.render()
step_images.append(rgb_array)
st.image(rgb_array, caption="Step 3: Down")

step_results.append(env.step(1)) #down
rgb_array = env.render()
step_images.append(rgb_array)
st.image(rgb_array, caption="Step 4: Down")

step_results.append(env.step(1)) #down
rgb_array = env.render()
step_images.append(rgb_array)
st.image(rgb_array, caption="Step 5: Down")

step_results.append(env.step(2)) #right
rgb_array = env.render()
step_images.append(rgb_array)
st.image(rgb_array, caption="Step 6: Right")

for i, result in enumerate(step_results):
    st.write(f"Step {i+1}: {result}")

snv=gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
action=snv.reset()
st.write(f"Reset action: {action}")
