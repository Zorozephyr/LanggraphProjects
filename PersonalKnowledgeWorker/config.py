"""
Configuration and setup for Personal Knowledge Worker
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import httpx
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv(override=True)

# Paths
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "amdocsKnowledgeBase"
COMPANY_MHT_PATH = KNOWLEDGE_BASE_DIR / "Company.mht"
IMAGES_CACHE_PATH = KNOWLEDGE_BASE_DIR / "images_cache.pkl"
VECTOR_DB_PATH = KNOWLEDGE_BASE_DIR / "knowledge_base_db"
EMBEDDINGS_MODEL_PATH = "C:/AgenticAI/all-MiniLM-L6-v2"

# Azure Configuration
AUTOX_API_KEY = os.getenv("AUTOX_API_KEY")
NTNET_USERNAME = (os.getenv("NTNET_USERNAME") or "").strip()

# Set proxy bypass
os.environ["NO_PROXY"] = ",".join(filter(None, [
    os.getenv("NO_PROXY", ""),
    ".autox.corp.amdocs.azr",
    "chat.autox.corp.amdocs.azr",
    "localhost",
    "127.0.0.1"
]))
os.environ["no_proxy"] = os.environ["NO_PROXY"]


def get_llm():
    """Initialize and return Azure ChatOpenAI instance"""
    http_client = httpx.Client(
        verify=r"C:\amdcerts.pem",
        timeout=30.0
    )
    
    async_http_client = httpx.AsyncClient(
        verify=r"C:\amdcerts.pem",
        timeout=30.0
    )
    
    return AzureChatOpenAI(
        azure_endpoint="https://chat.autox.corp.amdocs.azr/api/v1/proxy",
        api_key=AUTOX_API_KEY,
        azure_deployment="gpt-4o-128k",
        model="gpt-4o-128k",
        temperature=0.1,
        openai_api_version="2024-08-01-preview",
        default_headers={"username": NTNET_USERNAME, "application": "testing-proxyapi"},
        http_client=http_client,
        http_async_client=async_http_client
    )


def get_embeddings():
    """Initialize and return HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name=str(EMBEDDINGS_MODEL_PATH),
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

