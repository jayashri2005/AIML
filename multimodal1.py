from huggingface_hub import InferenceClient
import base64
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

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


llm=ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        task="image-text-to-text",
    )
)
print(llm.invoke(messages))