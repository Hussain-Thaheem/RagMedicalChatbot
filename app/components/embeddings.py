from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embeddings_model():
   try:
        logger.info("Loading HuggingFace embeddings model...")

        model = HuggingFaceEmbeddings(
         model_name="sentence-transformers/all-MiniLM-L6-v2",
      )

        logger.info("HuggingFace embeddings model loaded successfully.")
      
        return model
   except Exception as e:
      logger.error(f"Error loading HuggingFace embeddings model: {e}")
      raise CustomException(f"An error occurred while loading the embeddings model: {e}") from e
        