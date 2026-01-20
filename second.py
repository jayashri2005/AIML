import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()
os.getenv("HF_TOKEN")
import streamlit as st
llmhg=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))


st.title("HugggingFace LLM with LangChain")
user_input = st.text_input("Enter your prompt:")
if st.button("Get Response"):
    response = llmhg.invoke(user_input)
    st.write("Response from HuggingFace LLM :",response.content)