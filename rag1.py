""" Embedding 
in the form of numbers 
represent in the form of numbers 

eg: dog [2,3] cat[3,2]
rep in 2d form and the graphical rep of x,y axis -> vector
i.e angle b/w two axis 
cos theta -> theta 0 = 0[full similarity], 90=1 [0] 
dot matrix used b/w two matrix to find similarity [a.b=abcos]
bigger list of num = big properties

sentiment analysis = way of decorate data that can be easily understandable

classical 
extract unique words and rep 1 if present or 0 

nn - > extracted dat[1,0] feed into nn
neuron takes guess wrk in training [single nueron -re of data]
linear combination 


used via pytourch,sentenceTransformer use embedding depends on ai model
pytourch llry to numpy additional contain -> find gradiants

RAG -> retrival and generation retrive most match one in docx
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
#from sentence_transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np 
import faiss
import os, getpass 
os.environ['HF_TOKEN'] = getpass.getpass('Hugging Token:')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

def get_embedding(texts):
    inputs = tokenizer(texts, return_tensors="pt",padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][:,0,:]  
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings  

retriever_model = SentenceTransformer('bert-base-nli-mean-tokens')

documents = [
     "A sudden power outage plunged the entire building into darkness."
    "Employees gathered near the windows, trying to figure out what happened."
    "The emergency lights flickered on after a few seconds, revealing a strange humming sound in the hallway."
    "Curious but cautious, Riya stepped forward to investigate the source of the noise."

]

query ="What do you think Riya should do next to ensure everyones safety?"

document_embeddings = get_embedding(documents)
query_embedding = get_embedding([query])
document_embeddings.shape
torch.Size([3,768])

def cosine_similarity(embedding1,embedding2):
    return torch.nn.functional.cosine_similarity(embedding1,embedding2)

for doc_embedding in document_embeddings:
    print(cosine_similarity(query_embedding,doc_embedding))

similarities = [cosine_similarity(query_embedding,doc_embedding) for doc_embedding in document_embeddings]
print(similarities)

ranked_documents = sorted(zip(documents, similarities), key=lambda x:x[1], reverse=True)
print(ranked_documents)

top_documents = [doc for doc, _ in ranked_documents[:2]]
print(top_documents)

query + " [SEP] " + " " . join(top_documents)

augmented_input = query + " [SEP] " + " ".join(top_documents)

input_ids = tokenizer.encode(augmented_input, return_tensors="pt",padding=True,truncation=True)
print(input_ids)

outputs = model.generate(input_ids, max_length=150, num_beams=2, early_stopping=True)
print(outputs)

document_embeddings.shape
torch.Size([3,768])

index = faiss.IndexFlatL2(document_embeddings.shape[1])
print(index.is_trained)

index.add(document_embeddings.numpy())
print(index.ntotal)

#retrieve info
query_embedding = get_embedding([query])

distance,indices = index.search(query_embedding.detach().numpy(),k=5)
print(distance[0],indices[0])

top_documents = [documents[i] for i in indices[0]]
print(top_documents)

#to get minute details of doc -> chunk is used  chunk size bid -> good context buildup size=100 i get upto 20% overlap reduce space nd improve cony
#imp -> what chunk

def chunk_text(text, max_length=100):
    words=text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words),max_length)]
    return chunks

chunks=[]
for doc in documents:
    chunks.extend(chunk_text(doc, max_length=21))

print(chunks)

chunk_embeddings = get_embedding(chunks)
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])

index.add(chunk_embeddings.detach().numpy())
query_embedding=get_embedding([query])
distance,indices=index.search(query_embedding.detach().numpy(),k=7)
print(indices)
