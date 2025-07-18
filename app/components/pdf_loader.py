import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH,CHUNK_SIZE,CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    try:
     
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Directory {DATA_PATH} does not exist.")
        
        loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            raise CustomException("No PDF files found in the specified directory.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Loaded and split {len(chunks)} chunks from PDF files in {DATA_PATH}.")
        
        return chunks
    except Exception as e:
        logger.error(f"Error loading PDF files: {e}")
        raise CustomException(f"An error occurred while loading PDF files: {e}") from e
        return []