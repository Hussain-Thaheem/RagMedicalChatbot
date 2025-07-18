import os
from dotenv import load_dotenv
load_dotenv()


HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set. Please set it in your .env file.")

HUGGINGFACE_REPO_ID = os.environ.get("HUGGINGFACE_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.3")
if not HUGGINGFACE_REPO_ID:
    raise ValueError("HUGGINGFACE_REPO_ID environment variable is not set. Please set it in your .env file.")

DB_FAISS_PATH = os.environ.get("DB_FAISS_PATH", "vectorstore/db_faiss")
if not DB_FAISS_PATH:
    raise ValueError("DB_FAISS_PATH environment variable is not set. Please set it in your .env file.")

DATA_PATH = "data/"
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"Created data directory at {DATA_PATH}")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be a positive integer. Please set it in your .env file.")

CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
if CHUNK_OVERLAP < 0:
    raise ValueError("CHUNK_OVERLAP must be a non-negative integer. Please set it in your .env file.")

