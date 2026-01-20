#graph rag, agentic rag,fine tuning(dl,cv)
#two types of chattemplate
#message-humanai,userai
#chatprompttemplate->simple prompt

#types of praser -> typing(type validation-str,dtype),pydantic(data validation->condition of data),ordinary situation parser use string parser,json parser,using advanced llm->structured output parser

#recursivetextsplitter helps to iplement chunks,document arrnge them in doc/file format
#imp->emperical overlap with 20% of chunk size 

from langchain_core import embeddings, retrievers
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.tools import retriever
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os, getpass
import numpy as np
from openai import vector_stores


#from FirstProject.myenv.rag1 import chunks 
load_dotenv('../.env')
#ChatHuggingFace(model="microsoft/DialoGPT-medium", token=os.getenv("HF_TOKEN"))
os.environ['HF_TOKEN'] = getpass.getpass('Hugging Token:')

documents = [
     "A sudden power outage plunged the entire building into darkness."
    "Employees gathered near the windows, trying to figure out what happened."
    "The emergency lights flickered on after a few seconds, revealing a strange humming sound in the hallway."
    "Curious but cautious, Riya stepped forward to investigate the source of the noise."

]

prompt = ChatPromptTemplate.from_template("""
you are an assistant that answers questions strictly
using the provided context.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"I don't know based on the provided context."
""")
print(prompt)

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b"))
chain = prompt|llm
context = "\n".join(documents)
response = chain.invoke({
    "context": context,
    "question": "Who stepped forward to investigate the strange noise in the hallway?"
})
print(response.content)


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

documents_for_splitter = [Document(page_content=doc) for doc in documents]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=['\n\n',"\n"," ",""]

)
chunks=text_splitter.split_documents(documents_for_splitter)
print(chunks)

context ="\n\n".join(chunk.page_content for chunk in chunks)
print(context)

response = chain.invoke({
    "context": context,
    "question": "Describe what happened in the building immediately after the sudden power outage and how the employees reacted."
})
print(response.content)

#using embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
text_to_embed = "This is just simple text."
query_result = embeddings.embed_query(text_to_embed)
print(len(query_result))
print(query_result[:5])

vectorstore = FAISS.from_documents(chunks,embeddings)
retrievers = vectorstore.as_retriever()


prompt = ChatPromptTemplate.from_messages([
    ("system","Use the retrieved context to answer the user."),
    ("human","{question}\n\nContext:\n{context}")
])

chain = prompt|llm
history=[]
def ask(question):
    global history
    docs = retrievers.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    response=chain.invoke({"context":context,"question":question})
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response.content))
    return response.content 
print(ask("Who stepped forward to investigate the strange noise in the hallway?"))

texts = [
     "A sudden power outage plunged the entire building into darkness."
    "Employees gathered near the windows, trying to figure out what happened."
    "The emergency lights flickered on after a few seconds, revealing a strange humming sound in the hallway."
    "Curious but cautious, Riya stepped forward to investigate the source of the noise."

]

doc_embedding=np.array(embeddings.embed_documents(texts))
user_query = "Who stepped forward to investigate the strange noise in the hallway?"
doc_vectors = np.array(embeddings.embed_documents(texts))
query_vector = np.array(embeddings.embed_query(user_query))

scores = [np.dot(query_vector,doc_vec) for doc_vec in doc_vectors]
print(scores)

best_index = np.argmax(scores) #return best index 

print(best_index)

print(documents[best_index])

docs = ["A sudden power outage plunged the entire building into darkness.Employees gathered near the windows, trying to figure out what happened.The emergency lights flickered on after a few seconds, revealing a strange humming sound in the hallway.Curious but cautious, Riya stepped forward to investigate the source of the noise."]
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.create_documents(docs)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")
vectordb = FAISS.from_documents(chunks, emb)
retrievers = vectordb.as_retriever()
prompt = ChatPromptTemplate.from_template("""
you are an assistant that answers questions strictly
using the provided context.

Context:
{context}

Question:
{question}

Answer;
""")
print(prompt)

def get_answer(query):
    context_docs = retrievers.invoke(query)
    context = "\n\n".join([d.page_content for d in context_docs])

    chain = prompt | llm
    return chain.invoke({"context":context,"question":query})

print(get_answer("Who stepped forward to investigate the strange noise in the hallway?"))

