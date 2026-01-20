import streamlit as st
from human import get_history, send_message

st.set_page_config(page_title="Chat with HF LLM", layout="centered")
st.title("Chat — HuggingFace LLM (via langchain_core)")

# Session id to separate histories
session_id = st.text_input("Session id", value="harry")

# Simple connection: import backend directly; let errors surface normally

# Show existing history messages
if "history_display" not in st.session_state:
    st.session_state.history_display = []

# Load the history for display
history = get_history(session_id)
msgs = history.messages

st.subheader("Conversation history")
if msgs:
    for m in msgs:
        role = type(m).__name__
        st.write(f"**{role}**: {m.content}")
else:
    st.write("(no history yet)")

# User input
user_input = st.text_input("Your message", key="user_input")
if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        resp = send_message(session_id, user_input)
        content = getattr(resp, "content", str(resp))
        st.success(content)
        # Refresh display of history
        st.experimental_rerun()


