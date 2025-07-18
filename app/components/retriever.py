from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate  # Corrected import
from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from typing import Optional

logger = get_logger(__name__)

# Define the custom prompt template outside to prevent recreating on every call
CUSTOM_PROMPT_TEMPLATE = """
Answer the following medical questions in 2-3 lines maximum using only the information provided in the context.
If the answer is not present in the context, say "I don't know".

context: {context}
question: {question}
answer:
"""

def set_custom_prompt() -> PromptTemplate:
    """
    Set a custom prompt template for the RetrievalQA chain.
    """
    try:
        logger.info("Setting custom prompt template for RetrievalQA chain...")
        return PromptTemplate(
            input_variables=["context", "question"],
            template=CUSTOM_PROMPT_TEMPLATE
        )
    except Exception as e:
        logger.error(f"Error setting custom prompt: {e}")
        raise CustomException(f"An error occurred while setting the custom prompt: {e}") from e


def create_qa_chain() -> Optional[RetrievalQA]:
    try:
        logger.info("Creating RetrievalQA chain...")

        # Load the vector store
        vector_store = load_vector_store()
        if vector_store is None:
            logger.error("Vector store could not be loaded. Ensure it exists and is accessible.")
            raise CustomException("Vector store could not be loaded.")
        
        logger.info("Vector store loaded successfully.")
        
        # Load the LLM
        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN)
        if llm is None:
            logger.error("LLM could not be loaded. Ensure the Hugging Face repo ID and token are correct.")
            raise CustomException("LLM could not be loaded.")
        
        logger.info("LLM loaded successfully.")
        
        # Set the custom prompt
        custom_prompt = set_custom_prompt()
        
        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # 'stuff' might be fine, but could change to another method if needed
            retriever=vector_store.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=False,  # Set to True if you want to return source documents
            chain_type_kwargs={"prompt": custom_prompt}
        )
        logger.info("RetrievalQA chain created successfully.")
        return qa_chain

    except CustomException as ce:
        logger.error(f"CustomException occurred: {ce}")
        raise ce  # Re-raise CustomException for further handling
    except Exception as e:
        logger.error(f"Error creating RetrievalQA chain: {e}")
        raise CustomException(f"An error occurred while creating the RetrievalQA chain: {e}") from e
