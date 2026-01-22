from langchain_core.tools import retriever  
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts([
    "Cross encoders score query-document pairs",
    "FAISS is a fast vector search library",
    "LangChain changed retriever import after 1.0",
    "Compression is different from reranking"
], embedding=embeddings)
retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
# print("Testing retriever:")
# retriever_docs = retriever.invoke("How does cross encoding work?")
# for doc in retriever_docs:
#     print(f"- {doc.page_content}")
reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

def retrieve_and_rerank(query,top_n=5):
    docs=retriever.invoke(query)
    pairs=[(query,d.page_content) for d in docs]
    scores=reranker.predict(pairs)
    reranked=sorted(zip(scores,docs),key=lambda x:x[0],reverse=True)
    return [doc for _,doc in reranked[:top_n]]

docs = retrieve_and_rerank("How does cross encoding work?")
for d in docs:
    print(d.page_content)