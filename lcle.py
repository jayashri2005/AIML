
# lcle.py
import os
from dotenv import load_dotenv

# ---- Load .env and set both vars ----
load_dotenv()  # loads HF_TOKEN / HUGGINGFACEHUB_API_TOKEN from .env
hf = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf:
    raise RuntimeError("HF token not found. Add it to .env as HF_TOKEN or HUGGINGFACEHUB_API_TOKEN")

# Set the canonical var expected by LangChain integrations
os.environ["HF_TOKEN"] = hf

# ---- LangChain (pure HuggingFace) ----
from langchain_huggingface import HuggingFaceEndpoint   # NEW package (no deprecation)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Choose a chat/instruct model that supports text-generation via HF providers
# You can change this to another instruct model available via HF Inference Providers
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    temperature=0.0,
    max_new_tokens=200,
    # provider="auto",  # optional; HF router will pick a provider
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one sentence, formatted as a JSON object with a single key named 'explanation'."
)

jsparser = JsonOutputParser()
chain = prompt | llm | jsparser

result = chain.invoke({"topic": "LCEL in LangChain"})
print(result)
