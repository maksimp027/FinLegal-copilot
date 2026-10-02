import logging
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ingestion import PipelineConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def search_policies(query: str, config: PipelineConfig, top_k: int = 2) -> List[Tuple[Document, float]]:
    logger.info(f"Searching for query: '{query}'")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

    client = QdrantClient(path=config.db_path)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
        embedding=embeddings,
    )

    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results


if __name__ == "__main__":
    cfg = PipelineConfig()

    test_complaint = "I was scammed over the phone and transferred money to a fraudster. Can I get a refund?"
    found_docs = search_policies(query=test_complaint, config=cfg, top_k=2)

    for i, (doc, score) in enumerate(found_docs, 1):
        print(f"\n[Result {i}] Score (Relevance): {score:.4f}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Text: {doc.page_content}")