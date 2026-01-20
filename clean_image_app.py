import streamlit as st
import base64
import os
from dotenv import load_dotenv
import requests
from PIL import Image
import io
from huggingface_hub import InferenceClient

load_dotenv()
api_key = os.getenv("HF_TOKEN")

def generate_image_from_text(text_description):
    try:
        # Use HuggingFace InferenceClient
        client = InferenceClient(token=api_key)
        
        # Generate image using client
        image_bytes = client.text_to_image(
            prompt=text_description,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image_bytes.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        result = f"data:image/png;base64,{img_base64}"
        return result
        
    except Exception as e:
        # Fallback to direct API call if client fails
        try:
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            response = requests.post(API_URL, headers=headers, json={"inputs": text_description})
            
            if response.status_code != 200:
                # Try to new router API
                API_URL_NEW = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
                response = requests.post(API_URL_NEW, headers=headers, json={"inputs": text_description})
                
                if response.status_code != 200:
                    raise Exception(f"Both APIs failed. Last error: {response.text}")
            
            # Convert response to image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            result = f"data:image/png;base64,{img_base64}"
            return result
            
        except Exception as fallback_error:
            raise Exception(f"Failed to generate image. Client error: {str(e)}. Fallback error: {str(fallback_error)}")

# Streamlit App
st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="wide")

st.title("🎨 AI Text-to-Image Generator")
st.markdown("Generate stunning images from your text descriptions using Hugging Face's Stable Diffusion XL")

# Initialize session state
if 'generated_image_url' not in st.session_state:
    st.session_state.generated_image_url = None

# Text input for image description
prompt = st.text_area(
    "Enter your image description:",
    placeholder="A majestic lion standing on a rocky cliff at sunset, photorealistic, detailed, high resolution",
    height=100,
    help="Be as descriptive as possible for better results. Include subjects, style, lighting, colors, and composition."
)

# Generate button
if st.button("🎨 Generate Image", type="primary", use_container_width=True):
    if prompt.strip():
        with st.spinner("🎨 Generating your image with Stable Diffusion XL..."):
            try:
                # Generate image
                image_url = generate_image_from_text(prompt)
                st.session_state.generated_image_url = image_url
                st.session_state.last_prompt = prompt
                
                # Success message
                st.success("✅ Image generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating image: {str(e)}")
                if "HF_TOKEN" in str(e) or "Authorization" in str(e):
                    st.warning("⚠️ Please check your HF_TOKEN in .env file")
    else:
        st.error("❌ Please enter an image description")

# Display generated image
if st.session_state.get('generated_image_url'):
    st.markdown("---")
    st.subheader("🖼️ Generated Image")
    
    # Display image with download option
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.image(
            st.session_state.generated_image_url, 
            caption=f"Prompt: {st.session_state.get('last_prompt', '')}",
            use_container_width=True
        )
    
    with col2:
        # Create download button
        if st.session_state.generated_image_url.startswith('data:image'):
            # Convert base64 to bytes for download
            image_data = st.session_state.generated_image_url.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            st.download_button(
                label="📥 Download",
                data=image_bytes,
                file_name="generated_image.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.markdown(f"[📥 Download Image]({st.session_state.generated_image_url})")

# Clear button
if st.session_state.get('generated_image_url'):
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.generated_image_url = None
        st.session_state.last_prompt = None
        st.rerun()
