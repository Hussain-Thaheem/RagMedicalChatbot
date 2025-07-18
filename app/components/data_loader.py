import os

from app.components.pdf_loader import load_pdf_files #It is also sending the chunks to the vector store
from app.components.vector_store import save_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def process_and_store_pdfs():
    """
    Process PDF files from the configured data path, split them into chunks,
    and save them to the FAISS vector store.
    
    Raises:
        CustomException: If an error occurs during processing or saving.
    """
    try:
        logger.info("Starting PDF processing and vector store saving...")

        # Load and split PDF files into chunks
        text_chunks = load_pdf_files()
        logger.info(f"Loaded {len(text_chunks)} text chunks from PDF files.")

        if not text_chunks:
            raise CustomException("No text chunks were generated from the PDF files.")

        # Save the text chunks to the FAISS vector store
        vector_store = save_vector_store(text_chunks)

        logger.info("PDF processing and vector store saving completed successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error in process_and_store_pdfs: {e}")
        raise CustomException(f"An error occurred while processing PDFs: {e}") from e
    

if __name__ == "__main__":
    try:
        process_and_store_pdfs()
    except CustomException as ce:
        logger.error(f"CustomException: {ce}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
