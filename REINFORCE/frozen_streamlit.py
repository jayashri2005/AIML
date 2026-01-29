import streamlit as st
from turtle import right
import gymnasium as gym 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import io

st.title("FrozenLake Environment Visualization")

print(gym.envs.registry.keys())
env=gym.make("FrozenLake-v1",is_slippery=False, render_mode="rgb_array")
env.reset()
rgb_array = env.render()

st.image(rgb_array, caption="FrozenLake Environment")

st.write(f"Observation Space: {env.observation_space.n}")
st.write(f"Action Space: {env.action_space.n}")

st.subheader("Environment Steps as GIF")
step_results = []
step_images = []

# Initial state
rgb_array = env.render()
step_images.append(rgb_array)

step_results.append(env.step(2)) #right -2 
rgb_array = env.render()
step_images.append(rgb_array)

step_results.append(env.step(2)) #right
rgb_array = env.render()
step_images.append(rgb_array)

step_results.append(env.step(1)) #down
rgb_array = env.render()
step_images.append(rgb_array)

step_results.append(env.step(1)) #down
rgb_array = env.render()
step_images.append(rgb_array)

step_results.append(env.step(1)) #down
rgb_array = env.render()
step_images.append(rgb_array)

step_results.append(env.step(2)) #right
rgb_array = env.render()
step_images.append(rgb_array)

# Convert RGB arrays to PIL Images and create GIF
pil_images = []
for img_array in step_images:
    pil_img = Image.fromarray(img_array)
    pil_images.append(pil_img)

# Create GIF in memory
gif_buffer = io.BytesIO()
pil_images[0].save(gif_buffer, format='GIF', save_all=True, append_images=pil_images[1:], duration=500, loop=0)
gif_buffer.seek(0)

# Display GIF
st.image(gif_buffer, caption="FrozenLake Steps Animation")

# Show step results
st.subheader("Step Results")
for i, result in enumerate(step_results):
    st.write(f"Step {i+1}: {result}")

snv=gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
action=snv.reset()
st.write(f"Reset action: {action}")
