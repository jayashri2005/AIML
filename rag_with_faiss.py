from huggingface_hub import InferenceClient
import base64
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint #llm
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer #text generatation purpose
import faiss #vector db
import numpy as np
from typing import List, Tuple
import streamlit as st 


load_dotenv()
api_key = os.getenv("HF_TOKEN")

class FAISSRAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = []
        self.index = None
        self.dimension = 384  
    
    def add_document(self, text: str, metadata: dict = None):
        embedding = self.embedding_model.encode(text, show_progress_bar=False)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(np.array([embedding]).astype('float32'))
        self.documents.append({
            'text': text,
            'metadata': metadata or {}
        })
        self.embeddings.append(embedding)
        
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float, dict]]:
        """Search for similar documents"""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx]['text'],
                    float(dist),
                    self.documents[idx]['metadata']
                ))
        
        return results
    
    def save_index(self, filepath: str):
        if self.index is not None:
            faiss.write_index(self.index, filepath)
            print(f"Saved FAISS index to {filepath}")
    
    def load_index(self, filepath: str):
        if os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            print(f"Loaded FAISS index from {filepath}")

rag = FAISSRAG()

image_path = r"C:\Emphasis\FirstProject\bird.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

img_base64 = encode_image(image_path)

client = InferenceClient(api_key=api_key)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            },
            {
                "type": "text",
                "text": "What breed is this bird? Describe its characteristics and habitat."
            }
        ]
    }
]

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        task="image-text-to-text",
    )
)

response = llm.invoke(messages)
print("Vision model response:", response)

rag.add_document(
    text=response.content,
    metadata={
        "image_path": image_path,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "type": "bird_analysis"
    }
)

additional_info = [
    "Sparrows are small birds commonly found in urban areas. They eat seeds and insects.",
    "Blue birds are known for their distinctive blue coloration and are often found in North America.",
    "Birds with white eye rings are typically small songbirds that live in wooded areas."
]

for info in additional_info:
    rag.add_document(
        text=info,
        metadata={"type": "general_bird_info"}
    )

rag.save_index("bird_knowledge.faiss")

def answer_question(question: str):
    """Answer question using RAG"""
    print(f"\n Question: {question}")
    
    relevant_docs = rag.search(question, k=2)
    
    if not relevant_docs:
        return "I don't have relevant information to answer this question."
    
    context = "\n".join([f"- {doc[0]}" for doc in relevant_docs])
    
    rag_messages = [
        SystemMessage(content="""You are a helpful assistant that answers questions about birds based on the provided context. 
        Use only the information from the context to answer. If the context doesn't contain the answer, say you don't have enough information."""),
        HumanMessage(content=f"""Context:
{context}

Question: {question}

Answer:""")
    ]
    
    answer = llm.invoke(rag_messages)
    return answer.content

questions = [
    "What bird is in the image?",
    "What are the characteristics of this bird?",
    "Where do birds with white eye rings live?",
    "What do sparrows eat?"
]

for question in questions:
    answer = answer_question(question)
    print(f"Answer: {answer}")

