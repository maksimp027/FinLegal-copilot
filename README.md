# FinLegal Copilot

FinLegal Copilot is an intelligent, multi-agent Retrieval-Augmented Generation (RAG) system built to automate the processing of customer complaints for financial and legal institutions. 

Powered by **LangGraph** and **FastAPI**, this system ensures high compliance by grounding all LLM responses strictly in corporate policies, effectively eliminating hallucinations. To optimize performance and reduce API costs, the system features a robust **Semantic Caching** layer.

---

## Key Features

- **Multi-Agent Architecture**: Utilizes LangGraph to implement a "Drafter-Critic" workflow. A Drafter agent generates the initial response, and a strict Critic (Compliance Officer) agent evaluates it against rigid rules.
- **Semantic Caching**: Incoming requests are vectorized and compared against previously resolved complaints in Qdrant. If a highly similar query is found (≥ 95% cosine similarity), the cached response is returned instantly without hitting the LLM API.
- **Low-Code Philosophy**: Business rules, policies, and compliance filters are completely decoupled from the codebase. They are injected dynamically from text files, allowing business users to update constraints without modifying Python code.
- **Production-Ready API**: Wrapped in a fast, asynchronous REST API using FastAPI.
- **Containerized**: Fully containerized using Docker and Docker Compose for easy deployment.
- **Automated Testing**: Comprehensive mock-based test suite using `pytest`.

## Architecture & Design Patterns

This project adheres to **DRY** (Don't Repeat Yourself) and **DDIA** (Designing Data-Intensive Applications) principles. A single source of truth (`PipelineConfig` built with Pydantic) dynamically manages environment variables, models, and database connections across all system layers.

### System Components:
1. **API Layer (`api.py`)**: Asynchronous FastAPI endpoints that handle incoming requests, query the Qdrant semantic cache, and asynchronously invoke the LangGraph workflow on cache misses.
2. **Ingestion Pipeline (`ingestion.py`)**: An ETL pipeline that reads raw policy documents (`data/policies.txt`), chunks them, generates dense vector embeddings (via `BAAI/bge-small-en-v1.5`), and stores them in Qdrant.
3. **Reasoning Graph (`graph.py`)**: 
   - **Retriever**: Queries the vector database for relevant legal context.
   - **Drafter**: Generates an empathetic yet legally strict response.
   - **Critic**: Analyzes the draft against compliance constraints (e.g., "Do not promise refunds"). If the draft fails, it loops back to the Drafter with actionable feedback.

## Tech Stack

- **Frameworks**: FastAPI, Uvicorn, LangChain, LangGraph
- **Machine Learning / LLMs**: Groq API (LLaMA 3 8B), HuggingFace Embeddings
- **Vector Database**: Qdrant (used for both RAG context and Semantic Caching)
- **Infrastructure & Testing**: Docker, Docker Compose, Pytest, Pydantic

---

## Quick Start

### 1. Prerequisites
- Docker and Docker Compose installed
- A valid Groq API Key

### 2. Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
CHUNK_SIZE=500
CHUNK_OVERLAP=50
COLLECTION_NAME=finlegal_policies
DB_PATH=./local_qdrant
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=llama3-8b-8192
```

### 3. Provide Business Logic
Place your policies and rules in the `data/` directory (these files are intentionally ignored by git to protect private data):
- `data/policies.txt`: Corporate rules, SLAs, and legal guidelines.
- `data/filters.txt`: Strict conditions for the Critic node (e.g., "1. Do not promise refunds. 2. Only use provided context.")

### 4. Run with Docker (Recommended)
Launch the application:
```bash
docker compose up --build -d
```

Initialize the vector database with your policies (run this once):
```bash
docker exec -it finlegal_copilot_api python ingestion.py
```

The API will now be available at `http://localhost:8000/docs`.

---

## Running Locally (Without Docker)

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Populate the vector database:
   ```bash
   python ingestion.py
   ```
3. Start the API server:
   ```bash
   uvicorn api:app --reload
   ```

---

## Testing

The project includes an automated test suite that mocks the Qdrant database and the Groq LLM API to ensure safe, cost-free testing.

Run the tests using `pytest`:
```bash
PYTHONPATH=. pytest tests/
```
