import streamlit as st
import base64
import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
import tempfile

# Suppress model loading messages
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

load_dotenv()
api_key = os.getenv("HF_TOKEN")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

def analyze_image(image_path, question):
    with SuppressOutput():
        img_base64 = encode_image(image_path)
        
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
                        "text": question
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
        return response.content

# Streamlit App
st.set_page_config(page_title="Image Analyzer")

st.title("Image Analyzer")

# Initialize session state
if 'temp_image_path' not in st.session_state:
    st.session_state.temp_image_path = None

# Main content
st.header(" Upload Image")

uploaded_file = st.file_uploader(
    "Choose a bird image...",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=400)
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state.temp_image_path = tmp_file.name
    
    # Question input
    question = st.text_area(
        "Enter your question:",
        value="What breed is this bird? Describe its characteristics and habitat.",
        height=100
    )
    
    if st.button(" Get Response", type="primary"):
        if st.session_state.temp_image_path:
            with st.spinner("Analyzing image..."):
                try:
                    answer = analyze_image(st.session_state.temp_image_path, question)
                    
                    st.subheader(" Answer:")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Please upload an image first")


