from fastapi import FastAPI
from pydantic import BaseModel
import time
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from graph import build_graph

from ingestion import PipelineConfig

# Module-level singletons (initialized on server startup)
config = PipelineConfig()
embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
qdrant = QdrantClient(path=config.db_path)

# Ensure semantic_cache collection exists
try:
    qdrant.get_collection("semantic_cache")
except Exception:
    qdrant.create_collection(
        collection_name="semantic_cache",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

workflow_app = build_graph()
app = FastAPI(title="FinLegal Copilot API", description="FastAPI with Semantic Caching")

class ComplaintRequest(BaseModel):
    text: str

class ComplaintResponse(BaseModel):
    response: str
    source: str
    cached: bool
    latency_ms: float

@app.post("/api/v1/complaints/resolve", response_model=ComplaintResponse)
async def resolve_complaint(req: ComplaintRequest):
    start_time = time.time()
    
    # 1. Semantic Caching Layer
    vector = embeddings.embed_query(req.text)
    search_results = qdrant.search(
        collection_name="semantic_cache",
        query_vector=vector,
        limit=1,
        score_threshold=0.95
    )
    
    if search_results:
        return ComplaintResponse(
            response=search_results[0].payload.get("response", ""),
            source="Semantic Cache",
            cached=True,
            latency_ms=(time.time() - start_time) * 1000
        )
        
    # 2. Cache Miss: Execute LangGraph asynchronously
    initial_state = {"complaint": req.text}
    result = await workflow_app.ainvoke(initial_state)
    response_text = result["draft_response"]
    
    # 3. Store the new complaint and generated response into the Semantic Cache
    qdrant.upsert(
        collection_name="semantic_cache",
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"response": response_text, "complaint": req.text}
            )
        ]
    )
    
    return ComplaintResponse(
        response=response_text,
        source="LangGraph Generation",
        cached=False,
        latency_ms=(time.time() - start_time) * 1000
    )
