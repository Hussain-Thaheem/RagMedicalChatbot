from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN):
    try:
        logger.info("Loading LLM from Hugging Face Hub...")

        # Use conversational task here
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            temperature=0.3,
            max_new_tokens=512,
            return_full_text=False,
            task="conversational"  # Set task as 'conversational'
        )

        logger.info("LLM loaded successfully.")
        return llm
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise CustomException(f"An error occurred while loading the LLM: {e}") from e
