import pytest
import os
from unittest.mock import patch, MagicMock

# Set required environment variables before importing the FastAPI app
os.environ["GROQ_API_KEY"] = "mock_key"
os.environ["EMBEDDING_MODEL"] = "BAAI/bge-small-en-v1.5"

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

@patch("api.qdrant")
@patch("api.workflow_app")
def test_resolve_complaint_cache_miss(mock_workflow_app, mock_qdrant):
    # Mock a cache miss (empty list from Qdrant)
    mock_qdrant.search.return_value = []
    
    # Mock LangGraph response for ainvoke
    async def mock_ainvoke(*args, **kwargs):
        return {"draft_response": "This is a drafted response from the LLM."}
    
    mock_workflow_app.ainvoke.side_effect = mock_ainvoke
    
    # Mock Qdrant upsert to not throw errors
    mock_qdrant.upsert.return_value = None
    
    response = client.post(
        "/api/v1/complaints/resolve",
        json={"text": "My account was charged twice."}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "LangGraph Generation"
    assert data["cached"] is False
    assert data["response"] == "This is a drafted response from the LLM."
    # Verify qdrant.upsert was called to cache the result
    assert mock_qdrant.upsert.called

@patch("api.qdrant")
def test_resolve_complaint_cache_hit(mock_qdrant):
    # Mock a cache hit
    mock_hit = MagicMock()
    mock_hit.payload = {"response": "This is a cached response."}
    mock_qdrant.search.return_value = [mock_hit]
    
    response = client.post(
        "/api/v1/complaints/resolve",
        json={"text": "My account was charged twice."}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "Semantic Cache"
    assert data["cached"] is True
    assert data["response"] == "This is a cached response."
