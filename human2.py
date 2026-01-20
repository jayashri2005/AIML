print("human2: script start")
import sys
sys.stdout.flush()

#sate management to streamlit 

#State management

import os
from dotenv import load_dotenv
import streamlit as st
import traceback
load_dotenv()
os.getenv("HF_TOKEN")
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableWithMessageHistory

llmhg=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))

prompt = ChatPromptTemplate.from_messages([
    ("system", "Please refine you answer according to example given"),
    MessagesPlaceholder(variable_name="history"),
    #("human","{user_input}")
    #("human", "what is spoken in Russia?")
    ("human", "{input}")
    ])

store={}
def get_history(session_id: str):  #key: session identifier
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


app=RunnableWithMessageHistory(
    runnable= prompt | llmhg,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
)
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

# Configure LLM (uses env for token if required)
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


# Wrap the prompt + llm in a runnable that manages history
app = RunnableWithMessageHistory(
    runnable=prompt | llmhg,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
)


st.set_page_config(page_title="Chat (human2)", layout="centered")
st.title("Chat — HuggingFace LLM (human2)")

session_id = st.text_input("Session id", value="harry")

st.subheader("Conversation")
history = get_history(session_id)
if history.messages:
    for m in history.messages:
        role = type(m).__name__
        st.write(f"**{role}**: {m.content}")
else:
    st.write("(no history yet)")

user_input = st.text_input("Your message", key="user_input")
if st.button("Send"):
    # Directly call the runnable and display the response
    resp = app.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
    content = getattr(resp, "content", str(resp))
    st.success(content)
    # Re-fetch and display updated history in the same run (experimental_rerun
    # may not be available in some Streamlit versions).
    history = get_history(session_id)
    st.subheader("Updated conversation")
    if history.messages:
        for m in history.messages:
            role = type(m).__name__
            st.write(f"**{role}**: {m.content}")
    else:
        st.write("(no history yet)")
    # Clear the input box for convenience
    try:
        st.session_state["user_input"] = ""
    except Exception:
        # older Streamlit may not allow direct session_state manipulation here
        pass
