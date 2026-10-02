import os
import logging
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pydantic_settings import BaseSettings, SettingsConfigDict

class PipelineConfig(BaseSettings):
    """
    Pydantic model for configuration.
    Reads automatically from .env
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: str = Field(default="", description="API key for Groq")
    chunk_size: int = Field(default=500, description="Maximum number of characters in a text chunk")
    chunk_overlap: int = Field(default=50, description="Number of overlapping characters between chunks")
    collection_name: str = Field(default="finlegal_policies", description="Collection name in Qdrant")
    db_path: str = Field(default="./local_qdrant", description="Path to the local Qdrant database")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="Model used for generating embeddings")
    llm_model: str = Field(default="llama3-8b-8192", description="LLM model used for text generation")

from langchain_community.document_loaders import DirectoryLoader, TextLoader

def get_documents(data_dir: str = "data") -> List[Document]:
    """
    Loads documents from the specified directory using LangChain's DirectoryLoader.
    """
    logger.info(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="policies*.txt", loader_cls=TextLoader)
    return loader.load()


def run_ingestion_pipeline(config: PipelineConfig):
    try:
        logger.info("Starting the Data Ingestion pipeline...")

        # Load data
        documents = get_documents()
        logger.info(f"Loaded {len(documents)} document(s).")

        # Chunking (Text Splitting)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Documents split into {len(chunks)} chunks.")

        # Using BGE (one of the best open-source models for semantic search)
        embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

        # Using local Qdrant (saves vectors in a folder)
        client = QdrantClient(path=config.db_path)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=embeddings,
        )

        # Upload our text chunks to the database
        vector_store.add_documents(chunks)
        logger.info(f"Successfully saved vectors to Qdrant at: {config.db_path}")

    except Exception as e:
        logger.error(f"Error in the Ingestion pipeline: {e}")
        raise