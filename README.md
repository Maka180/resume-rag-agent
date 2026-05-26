# IntellectRAG — AI Recruitment Intelligence Agent

IntellectRAG is an intelligent, production-ready Retrieval-Augmented Generation (RAG) system engineered to parse, vectorize, and audit unstructured curriculum vitae (CV) documents and candidate profiles. Utilizing a modern stack powered by **FastAPI**, **LangChain**, and **MongoDB Atlas Vector Search**, the engine allows recruiters to ingest multiple candidate datasets concurrently and instantly execute targeted screening queries against a grounded knowledge base.

---

## 🚀 Key Features

* **Concurrent Multi-Document Ingestion**: Seamlessly handles batch uploads of text and PDF profiles via standard multi-part payload pipelines.
* **Vector Embeddings & Storage**: Leverages LangChain text splitters to chunk document layers and securely maps high-dimensional text matrices into MongoDB Atlas via Hugging Face.
* **Grounded Context Auditing**: Backed by high-performance LLM routing (via Groq LLaMA models) to perform zero-memory context audits, returning precise factual summaries alongside source citations.
* **Ultra-Responsive UI**: A modern front-end layout designed with Tailwind CSS, built with a mobile-first philosophy that perfectly stacks grid panels and guarantees `break-words` safety on mobile viewports.
* **Compliance Actions**: Includes automated features to clear runtime session contexts or export evaluations directly to structured CSV metrics.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI (Python 3.10+)
* **Orchestration Layer:** LangChain
* **Database & Vector Index:** MongoDB Atlas
* **Inference Engine:** Groq API (LLaMA-3 models)
* **Embedding Engine:** Hugging Face Transformers (`all-MiniLM-L6-v2`)
* **Frontend:** HTML5, JavaScript (ES6+), Tailwind CSS Architecture

---

## 📦 Project Structure

```text
├── main.py              # FastAPI server entry point, CORS configuration, & route mechanics
├── CoreRAG.py           # Embeddings, Vector Search indexing, & LangChain pipeline logic
├── requirements.txt     # Python runtime dependencies
├── templates/
│   └── index.html       # Mobile-optimized web control dashboard interface
└── README.md            # System deployment document# resume-rag-agent

⚙️ Installation & Local Setup
1. Clone the Repository
git clone [https://github.com/Maka180/resume-rag-agent.git](https://github.com/Maka180/resume-rag-agent.git)
cd resume-rag-agent

2. Configure Environment Variables
Create a .env file in the root directory and populate your infrastructure secrets:
MONGO_URI="your_mongodb_atlas_connection_string"
GROQ_API_KEY="your_groq_llama3_secret_token"

3. Install Dependencies
pip install -r requirements.txt

4. Initialize Local Runtime
Fire up the local developmental server via uvicorn:
uvicorn main:app --reload
Once initialized, visit your live engine dashboard at http://127.0.0.1:8000 or view your auto-generated documentation schemas at http://127.0.0.1:8000/docs
Method,Endpoint,Description,Payload Signature
POST,/upload,"Ingests multi-part files, chunks content, and indexes vector blocks.",files: UploadFile[]
GET,/ask,Queries grounded vector paths for candidate verification.,?question=string
POST,/clear_chat,Clears local cache context arrays from active operational states.,None
GET,/export,Downloads processed evaluations in structured CSV format.,None**

📱 Front-End Mobile Optimization Details
The associated index.html interface features deep optimizations for standard mobile viewports:

Fluid Grid Adapters: Transforms heavy desktop horizontal partitions (grid-cols-5) to single-stacked vertical flow-containers (grid-cols-1) beneath 768px breakpoints.

Container Defenses: Uses string overflow truncation and strict class rules to handle heavy embedding text blocks or massive grounding context outputs without visual horizontal breakages.

Touch Target Controls: Action links and submit buttons scale out dynamically to full width (w-full sm:w-auto) on small screens for easy tap execution.

📄 License
This repository is configured for academic, portfolio, and freelance evaluation workflows. All rights reserved. Built with precision in 2026.
