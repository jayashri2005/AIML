import streamlit as st
from turtle import right
import gymnasium as gym 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np 

# Configure page
st.set_page_config(page_title="FrozenLake Interactive", layout="wide")
st.title("🎮 FrozenLake Interactive Environment")

# Environment settings
st.markdown("---")
col1, col2 = st.columns([1, 3])
with col1:
    is_slippery = st.checkbox("🧊 Slippery Mode", value=False, help="When enabled, the agent may slip to adjacent cells with 1/3 probability")
with col2:
    st.write("")  # Empty space for alignment

st.markdown("---")

print(gym.envs.registry.keys())
env=gym.make("FrozenLake-v1",is_slippery=is_slippery, render_mode="rgb_array")
env.reset()

# Initialize session state
if 'env_initialized' not in st.session_state or st.session_state.get('prev_slippery') != is_slippery:
    st.session_state.env_initialized = True
    st.session_state.prev_slippery = is_slippery
    st.session_state.step_count = 0
    st.session_state.total_reward = 0
    st.session_state.done = False
    st.session_state.step_history = []
    # Reset environment and get initial state
    obs, info = env.reset()
    st.session_state.env_state = obs

# Reset environment to current session state before rendering
env.reset()
# Restore current state by stepping through history to current position
current_obs, info = env.reset()
for step_data in st.session_state.step_history:
    if not st.session_state.done:
        current_obs, reward, done, truncated, info = env.step(step_data['action'])
        if done:
            break

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    # Environment display
    st.subheader("🏔️ Environment State")
    rgb_array = env.render()
    st.image(rgb_array, caption=f"Step {st.session_state.step_count}", width=400)

with col2:
    # Stats panel
    st.subheader("📊 Statistics")
    st.metric("Current State", st.session_state.env_state)
    st.metric("Total Reward", st.session_state.total_reward)
    st.metric("Steps Taken", st.session_state.step_count)
    
    # Environment info
    st.subheader("ℹ️ Environment Info")
    st.write(f"**Observation Space:** {env.observation_space.n}")
    st.write(f"**Action Space:** {env.action_space.n}")

# Controls section
st.markdown("---")
st.subheader("🎮 Controls")

if st.session_state.done:
    st.success(f"🎉 Episode finished! Final reward: {st.session_state.total_reward}")
    if st.button("🔄 Reset Environment", use_container_width=True):
        st.session_state.env_state = env.reset()[0]
        st.session_state.step_count = 0
        st.session_state.total_reward = 0
        st.session_state.done = False
        st.session_state.step_history = []
        st.rerun()
else:
    # Control buttons in grid layout
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    
    action_taken = None
    with col2:
        if st.button("⬆️ Up (3)", use_container_width=True):
            action_taken = 3
    with col1:
        if st.button("⬅️ Left (0)", use_container_width=True):
            action_taken = 0
    with col3:
        if st.button("➡️ Right (2)", use_container_width=True):
            action_taken = 2
    with col2:
        if st.button("⬇️ Down (1)", use_container_width=True):
            action_taken = 1
    with col5:
        if st.button("🎲 Random", use_container_width=True):
            action_taken = env.action_space.sample()

    if action_taken is not None:
        result = env.step(action_taken)
        st.session_state.env_state, reward, st.session_state.done, truncated, info = result
        st.session_state.total_reward += reward
        st.session_state.step_count += 1
        st.session_state.step_history.append({
            'step': st.session_state.step_count,
            'action': action_taken,
            'action_name': ['Left', 'Down', 'Right', 'Up'][action_taken],
            'reward': reward,
            'new_state': st.session_state.env_state
        })
        st.rerun()

# Step history
if st.session_state.step_history:
    st.markdown("---")
    st.subheader("📜 Step History")
    
    # Create a nice table for step history
    history_data = []
    for step in st.session_state.step_history:
        history_data.append({
            "Step": step['step'],
            "Action": f"{step['action_name']} ({step['action']})",
            "Reward": step['reward'],
            "New State": step['new_state']
        })
    
    st.dataframe(history_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Navigate the frozen lake to reach the goal! Avoid holes and reach the target in as few steps as possible.")

snv=gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
action=snv.reset()
# st.write(f"Reference reset action: {action}")  # Hidden for cleaner UI
