from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import *;
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace


import os, getpass
os.environ["HF_TOKEN"] = getpass.getpass() 
llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))

texts = [
    "LangChain helps developers build LLM applications.",
    "FAISS is used for vector similarity search.",
    "Chat history must be manually maintained in LangChain 1.1.",
    "Retrievers are used in RAG pipelines.",
    "OpenAI embeddings create vector representations."
]


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
db = FAISS.from_texts(texts,embeddings)
retrievers = db.as_retriever()

store={}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]

rag_prompt_with_history = ChatPromptTemplate.from_messages([
    ("system","use the retrieved context to answer the user."),
    MessagesPlaceholder("history"), 
    ("human","{question}\n\nContext:\n{context}")
])

def get_context_from_retriever(question_dict):
    docs = retrievers.invoke(question_dict["question"])
    return "\n".join([d.page_content for d in docs])

runnable = RunnablePassthrough.assign(context=get_context_from_retriever)

rag_chain_with_context=(
    runnable | rag_prompt_with_history
)

conversational_rag_chain_with_history = RunnableWithMessageHistory(
 rag_chain_with_context,
 get_session_history,
 input_messages_key="question",
 history_messages_key="history"

)

def ask_with_managed_history(question: str, session_id: str = "default_session"):
    response = conversational_rag_chain_with_history.invoke(
        {"question":question},
        config={
            "configurable": {"session_id":session_id}
        }
    )
    return response.messages[0].content 
store.clear()

print("User (session1): What is FAISS?")
print("AI (session1):", ask_with_managed_history("what is FAISS?", session_id="session1"))
print("\n User (session1): what did I ask earlier?")
print("AI (session1):",ask_with_managed_history("what did I ask earlier?",session_id="session1"))











"""
print("\nUser (session2): How does LangChain handle memory?")
print("AI (session2):",ask_with_managed_history("How does LangChain handle memory?",session_id=))
"""