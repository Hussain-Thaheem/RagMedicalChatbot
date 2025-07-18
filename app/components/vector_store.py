from langchain_community.vectorstores import FAISS
import os
from app.components.embeddings import get_embeddings_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)


def load_vector_store():
    try:
        logger.info("Loading FAISS vector store...")

        embeddings_model = get_embeddings_model()
        if os.path.exists(DB_FAISS_PATH): 
            logger.info("Loading existing vectorstore...")
            return FAISS.load_local(DB_FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)
        else:
            logger.info(f"Vector store path Not {DB_FAISS_PATH} exists, loading the vector store.")
            
    except Exception as e:
        logger.error(f"Error loading FAISS vector store: {e}")
        raise CustomException(f"An error occurred while loading the vector store: {e}") from e


def save_vector_store(text_chunks):
    try:
        logger.info("Saving FAISS vector store...")
        if not text_chunks:
            raise CustomException("No text chunks provided to save the vector store.")
        
        logger.info(f"Saving vector store to {DB_FAISS_PATH}...")
        embeddings_model = get_embeddings_model()
        vector_store = FAISS.from_documents(text_chunks, embeddings_model)
        if not os.path.exists(DB_FAISS_PATH):
            os.makedirs(DB_FAISS_PATH)
        
        vector_store.save_local(DB_FAISS_PATH)
        logger.info(f"Vector store saved successfully at {DB_FAISS_PATH}.")
        return vector_store
    except Exception as e:
        logger.error(f"Error saving FAISS vector store: {e}")
        raise CustomException(f"An error occurred while saving the vector store: {e}") from e