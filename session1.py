import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

# --- LLM & prompt setup (same as human2) ---
llmhg = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "Please refine you answer according to example given"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Simple in-memory store for histories
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Runnable that manages prompt + history
app = RunnableWithMessageHistory(
    runnable=prompt | llmhg,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Streamlit UI ---
st.set_page_config(page_title="Session + Chat", layout="wide")
st.title("Combined Session and Chat")

left, right = st.columns([1, 2])

with left:
    st.header("Session counter")
    # Numeric input and session-stored counter
    num = st.number_input("Start / set number:", value=0, step=1, key="start_num")
    if st.button("Initialize Counter"):
        st.session_state["counterr"] = int(num)
    if "counterr" not in st.session_state:
        st.session_state["counterr"] = 0
    if st.button("Increment Counter"):
        st.session_state["counterr"] = int(st.session_state["counterr"]) + 1
    st.write("Current counter:", st.session_state["counterr"])

with right:
    st.header("Chat with LLM")
    session_id = st.text_input("Chat session id:", value="harry", key="chat_session_id")

    st.subheader("Conversation")
    history = get_history(session_id)
    if history.messages:
        for m in history.messages:
            role = type(m).__name__
            st.write(f"**{role}**: {m.content}")
    else:
        st.write("(no history yet)")

    user_input = st.text_input("Your message", key="chat_user_input")
    if st.button("Send", key="send_btn"):
        resp = app.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        content = getattr(resp, "content", str(resp))
        st.success(content)
        # Show updated history below
        history = get_history(session_id)
        st.subheader("Updated conversation")
        if history.messages:
            for m in history.messages:
                role = type(m).__name__
                st.write(f"**{role}**: {m.content}")
        else:
            st.write("(no history yet)")
        st.session_state["chat_user_input"] = ""

