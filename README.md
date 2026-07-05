#  Multi-Agent Knowledge Engine

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-green.svg)](https://www.langchain.com/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red.svg)](https://qdrant.tech/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-purple.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production‑grade RAG system with multi‑agent orchestration, semantic chunking, vector search, and citation‑grounded answers.**

---

##  What This Does

This system ingests messy PDF documents, chunks them semantically, stores them in a vector database, and answers questions with **citations and grounding verification** — preventing hallucination.

| Feature | Description |
|---------|-------------|
|  **Document Ingestion** | Parses PDFs (and other formats) with PyMuPDF |
|  **Semantic Chunking** | Smart text splitting with overlap for context retention |
|  **Vector Search** | Qdrant with OpenAI embeddings (1536‑dim) |
|  **Multi‑Agent Orchestration** | Query Understanding → Retrieval → Verification → Answer |
|  **Citations** | Every answer includes source references |
|  **Grounding Verification** | Ensures answers are grounded in source documents |
|  **Web Interface** | Streamlit dashboard for easy interaction |
|  **REST API** | FastAPI endpoints for programmatic access |

---

##  Architecture


┌─────────────────────────────────────────────────────────────┐
│ Streamlit UI / FastAPI │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Agent Orchestrator (LangGraph) │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │
│ │ Query │ │ Retrieval │ │ Verification │ │
│ │ Understanding│─▶│ Agent │─▶│ Agent │ │
│ │ Agent │ │ │ │ │ │
│ └─────────────┘ └─────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Vector Database (Qdrant) │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Chunk 1 │ Chunk 2 │ Chunk 3 │ ... │ Chunk N │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Document Ingestion Pipeline │
│ PDFs → PyMuPDF → Semantic Chunking → Embedding → Qdrant │
└─────────────────────────────────────────────────────────────┘



---

##  Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.11+ | Core logic |
| **Agent Framework** | LangGraph | Stateful multi‑agent orchestration |
| **LLM** | OpenAI GPT‑4o‑mini | Query understanding, answer generation, verification |
| **Embeddings** | OpenAI text‑embedding‑ada‑002 | Vector representation of chunks |
| **Vector Database** | Qdrant | Vector storage and similarity search |
| **PDF Extraction** | PyMuPDF (fitz) | Text extraction from PDFs |
| **Chunking** | Semantic chunker with overlap | Context‑preserving text splitting |
| **Web UI** | Streamlit | Interactive dashboard |
| **API** | FastAPI | RESTful endpoints |
| **Validation** | Pydantic | Data validation and serialization |
| **Logging** | Custom rolling logger | Production‑grade logging |

---

##  Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key
- (Optional) Qdrant Cloud account

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/knowledge-engine.git
cd knowledge-engine

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your API keys


Start Qdrant (Docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant


Run the Dashboard
streamlit run dashboard.py --server.port 8503
Open http://localhost:8503 in your browser.


Run the API (Optional)
python api.py
Open http://localhost:8001/docs for Swagger UI.


Usage Guide
1. Upload a Document
Open the Streamlit dashboard

In the sidebar, click Upload PDF

Select a PDF file

Click Ingest Document

2. Ask a Question
In the main panel, enter your question

Click Search

View the answer with citations and confidence score

3. Review Sources
Expand the Sources section to see which chunks were used

Each source shows the relevant text excerpt and similarity score

4. Check Verification Notes
Expand Verification Notes to see grounding details

The system validates that the answer is supported by the source material


Sample Output
Query
"What are the top 5 most important books on this list?"

Answer
Based on the reading list, the most important books are:

"The Art of Problem Solving" by Richard Rusczyk

"Deep Learning" by Ian Goodfellow

"Design Patterns" by Erich Gamma

"The Pragmatic Programmer" by David Thomas

"Clean Code" by Robert C. Martin

Citations
Source	Similarity
Chunk 3	0.92
Chunk 1	0.87
Chunk 5	0.81
Verification Notes
✅ Grounded: The answer is supported by the source material

✅ All sources are from the uploaded document

✅ No hallucination detected


Configuration
Variable	  Description	Default
OPENAI_API_KEY	  OpenAI API key	(required)
QDRANT_URL	  Qdrant endpoint	http://localhost:6333
CHUNK_SIZE	  Characters per chunk	512
CHUNK_OVERLAP	  Overlap between chunks	50
TOP_K	  Number of chunks to retrieve	5
SIMILARITY_THRESHOLD	 Minimum similarity score	0.7


Running Tests
pytest tests/ -v --cov=. --cov-report=term


Docker Deployment
# Build the image
docker build -t knowledge-engine .

# Run the container
docker run -p 8503:8503 -e OPENAI_API_KEY=your-key knowledge-engine


Project Structure
knowledge-engine/
├── .env.example          # Template for secrets
├── .gitignore
├── README.md             # This file
├── requirements.txt
├── config.py             # Configuration
├── logger.py             # Production logging
├── models.py             # Pydantic models
├── chunking.py           # Semantic chunking
├── embedding.py          # OpenAI embeddings
├── vector_store.py       # Qdrant interface
├── ingest.py             # Document ingestion
├── orchestrator.py       # Multi‑agent orchestration
├── dashboard.py          # Streamlit UI
├── api.py                # FastAPI
├── agents/
│   ├── __init__.py
│   ├── query_agent.py    # Query understanding
│   ├── retrieval_agent.py # Retrieval logic
│   └── verification_agent.py # Grounding verification
├── tests/
│   └── test_retrieval.py
└── docs/
    └── architecture.md


    Future Enhancements
Add support for Word, Excel, and Markdown files

Implement multi‑modal RAG (images, tables)

Add conversation memory for follow‑up questions

Deploy to Streamlit Cloud with Qdrant Cloud

Add PDF summarization and key‑point extraction

 License
MIT – Feel free to use for your own projects!

Built as part of an AI/ML portfolio project.