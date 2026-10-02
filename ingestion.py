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

class PipelineConfig(BaseSettings):
    """
        Pydantic model for configuration.
        Allows easy integration with .env files in a production environment.
        """
    chunk_size: int = Field(default=500, description="Maximum number of characters in a text chunk")
    chunk_overlap: int = Field(default=50, description="Number of overlapping characters between chunks")
    collection_name: str = Field(default="finlegal_policies", description="Collection name in Qdrant")
    db_path: str = Field(default="./local_qdrant", description="Path to the local Qdrant database")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="Model used for generating embeddings")

def get_mock_policies() -> List[Document]:
    """
        Temporary function mocking document loading.
        In a real project, this would use a PDFLoader or read from Confluence/Notion.
    """
    text = """
        Policy 1: Chargeback Procedures.
        If a customer reports an unauthorized transaction within 30 days of its occurrence, the company must initiate a chargeback procedure. If the report is filed after 30 days, the liability falls entirely on the customer.

        Policy 2: Social Engineering and Fraud.
        The company does not refund funds if the customer voluntarily transferred them to scammers (e.g., under the influence of a phone scam). Refunds are only possible in the event of a compromise of the bank's internal systems.

        Policy 3: Complaint Resolution (SLA).
        The standard processing time for an official complaint is 14 business days. If law enforcement involvement is required, the processing time may be extended up to 45 days, and the customer must be notified via email.
        """
    return [Document(page_content=text, metadata={"source": "internal_wiki_v1"})]


def run_ingestion_pipeline(config: PipelineConfig):
    try:
        logger.info("Starting the Data Ingestion pipeline...")

        # Load data
        documents = get_mock_policies()
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