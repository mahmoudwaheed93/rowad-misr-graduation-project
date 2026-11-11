@'
# 🏡 Aqar-AI (عقار ذكي) — Egyptian Real Estate RAG & Agentic Portal

**Aqar-AI** is an end-to-end AI-powered real estate search, comparison, and reservation platform tailored for the Egyptian property market. It combines semantic vector retrieval (**Qdrant**), multi-source portal integration (**Property Finder Egypt**, **Nawy**, and **Aqarmap Egypt**), and autonomous agent orchestration (**LangGraph** + **LangChain**) behind a **FastAPI** backend and an interactive dark-mode web dashboard.

---

## ✨ Key Features

- **Multi-Source Property Ingestion & Validation:** Normalizes, validates, and deduplicates property listings (prices in EGP, area in sqm, bedrooms, bathrooms, payment plans, and compound amenities) across **Property Finder**, **Nawy**, and **Aqarmap**.
- **Hybrid Vector + Metadata Search (Qdrant):** Encodes property documents into 384-dimensional dense embeddings (`sentence-transformers/all-MiniLM-L6-v2` / `intfloat/multilingual-e5-small`) with structured payload filtering for price ranges, bedroom counts, and locations.
- **Stateful LangGraph AI Advisor:** Supports bilingual conversations (Arabic & English) with thread-level memory (`MemorySaver`) and autonomous tool-calling:
  - `search_egypt_properties`: Queries the Qdrant vector database alongside live portal listings.
  - `calculate_mortgage_installment`: Computes down payments (e.g., 10%) and monthly/quarterly installments over custom tenures.
  - `submit_property_purchase_order`: Executes instant unit reservations, persists buyer orders to `data/orders/purchase_orders.json`, and provides direct checkout/listing links.
- **Interactive 3-Column Dark Dashboard (`ui/index.html`):**
  - **Left Column:** Live 4-stage pipeline visualizer & real-time execution logs.
  - **Center Column:** Source-filterable property cards with interactive photo inspection modal, installment calculator, and 1-click reservation.
  - **Right Column:** Bilingual LangGraph chat assistant that dynamically syncs recommended properties to the central grid.

---

## 🏗️ System Architecture

```text
[ Property Finder | Nawy | Aqarmap ] + [ Kaggle CSV Dataset (39.7k+ rows) ]
                                  │
                                  ▼
                 1. LangGraph Ingestion & Validation
              (Normalize EGP • Validate • Deduplicate)
                                  │
                                  ▼
                 2. Embedding Model + Qdrant Vector DB
               (384-dim Vectors + Structured Metadata)
                                  │
                                  ▼
                 3. User Query Agent (LangGraph ReAct)
         (Semantic Search • Compare • Installments • Reserve)
                                  │
                                  ▼
                 4. FastAPI Backend + Web Portal UI
---

## 📂 Project Structure

```text
aqar-ai/
├── app/
│   ├── agents/
│   │   └── real_estate_agent.py   # LangGraph ReAct agent & tool orchestration
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints (/search, /rag, /chat)
│   ├── core/
│   │   └── config.py              # Pydantic settings & environment configuration
│   ├── memory/
│   │   └── session_memory.py      # LangGraph conversation thread checkpointer
│   ├── rag/
│   │   ├── ingestion.py           # Multi-source CSV cleaning & JSON serialization
│   │   ├── chunking.py            # LangChain Document preparation
│   │   ├── embeddings.py          # HuggingFace 384-dim embedding model loader
│   │   ├── vector_store.py        # Qdrant client & collection management
│   │   ├── indexing.py            # Batch vector upsert pipeline
│   │   ├── retrieval.py           # Semantic search + Qdrant metadata filters
│   │   └── generation.py          # LCEL RAG prompt & LLM chain
│   ├── schemas/
│   │   └── property.py            # Pydantic request/response models
│   ├── tools/
│   │   └── property_tools.py      # Agent tools (Search, Installments, Purchase Order)
│   └── main.py                    # FastAPI application entrypoint & UI server
├── data/
│   ├── raw/
│   │   └── properties.csv         # Raw dataset (Kaggle waddahali/real-estate-listings)
│   ├── processed/
│   │   └── property_documents.json
│   └── orders/
│       └── purchase_orders.json   # Generated buyer reservation records
├── ui/
│   └── index.html                 # 3-Column Dark Dashboard & Chatbot UI
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & Set Up Virtual Environment
```powershell
git clone [https://github.com/mahmoudwaheed93/rowad-misr-graduation-project.git](https://github.com/mahmoudwaheed93/rowad-misr-graduation-project.git)
cd rowad-misr-graduation-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables (`.env`)
Create a `.env` file in the project root:
```env
PROJECT_NAME=Aqar-AI
VERSION=0.1.0

# LLM Provider (OpenRouter / Groq / OpenAI / Local llama-server)
LLM_BASE_URL=[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)
LLM_API_KEY=your_api_key_here
LLM_MODEL=meta-llama/llama-3.3-70b-instruct

# Qdrant Vector Store
QDRANT_LOCAL_PATH=./qdrant_storage
QDRANT_COLLECTION=egypt_real_estate

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

### 3. Download Dataset & Build Qdrant Index
Download the [Real Estate Listings Dataset](https://www.kaggle.com/datasets/waddahali/real-estate-listings) and place it at `data/raw/properties.csv`, then run the indexing pipeline:
```powershell
python -m app.rag.indexing
```

### 4. Start the FastAPI Server & Web Portal
```powershell
python -m uvicorn app.main:app --reload --port 8000
```
- **Web Portal & AI Chatbot:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🛠️ Tech Stack

- **Data Processing:** Python, Pandas, BeautifulSoup4, HTTPX
- **Embeddings:** HuggingFace Sentence-Transformers (`384-dim`)
- **Vector Database:** Qdrant (`qdrant-client`, `langchain-qdrant`)
- **RAG & Agent Orchestration:** LangChain, LangGraph
- **LLM Inference:** OpenRouter / Groq / `llama.cpp` (`meta-llama/llama-3.3-70b-instruct`)
- **Backend & Frontend:** FastAPI, Uvicorn, TailwindCSS, Vanilla JS
