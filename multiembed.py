from huggingface_hub import InferenceClient
import base64
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
import os

# Suppress model loading messages
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv()
api_key = os.getenv("HF_TOKEN")

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
                "text": "What breed is this bird?"
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

# Get vision model response
response = llm.invoke(messages)
print("Vision model response:", response)

# Get embedding for the response (FIXED - removed show_progress_bar from init)
model = SentenceTransformer('all-MiniLM-L6-v2')
response_embedding = model.encode(response.content, show_progress_bar=False)

print(f"Response text: {response.content}")
print(f"Embedding shape: {response_embedding.shape}")
print(f"Embedding: {response_embedding}")
