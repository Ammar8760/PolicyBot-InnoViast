# PolicyBot — Enterprise HR & Knowledge Assistant ⚡

An enterprise-grade Retrieval-Augmented Generation (RAG) platform that queries official HR and company policy documentation to deliver grounded, source-backed responses with strict fallback boundaries and real-time citation tracking.

---

## 🎯 Project Overview & Core Features

* **Multi-Format Document Ingestion Engine**: Seamlessly parses, indexes, and extracts structured knowledge from `.md`, `.txt`, and `.pdf` policy manuals.
* **Deterministic Threshold Gate**: Intercepts queries prior to LLM generation to eliminate hallucinations and minimize API resource consumption on out-of-scope requests.
* **Granular Source Citation & Verification**: Dynamically exposes exact source files, chunk indices, similarity scores, and raw underlying context for full enterprise transparency.
* **Dual-Vector Store Architecture**: Switches on-the-fly between local vector storage (ChromaDB) for high-speed offline dev and cloud indexing (Pinecone) for ephemeral deployment environments.
* **Real-time Parametric Control Panel**: Live sidebar UI to dynamically calibrate Top-K context retrieval limits and cosine similarity score thresholds.

---

## 🛠️ Tech Stack & Architecture
* **Frontend UI Framework**: Streamlit (Python)
* **LLM Engine**: Google Gemini 2.5 Flash
* **Embedding Model**: Google `gemini-embedding-001` (768-dim)
* **Vector Databases**: ChromaDB (Local Dev) & Pinecone (Cloud Storage)
* **PDF Parser**: `pypdf` Text Extractor
* **Environment Control**: `python-dotenv` & Streamlit Secrets (`st.secrets`)

---

## 🧪 Benchmark Evaluation Matrix

Evaluated across **20 operational query sets** divided into 4 key categories:

| Query Category | Total Samples | Passed | Partial | Failed | Accuracy Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A — Direct Policy Retrieval** | 8 | 8 | 0 | 0 | **100%** |
| **B — Multi-hop Reasoning** | 5 | 5 | 0 | 0 | **100%** |
| **C — Out-of-Scope Queries** | 4 | 4 | 0 | 0 | **100%** |
| **D — Ambiguous / Partial** | 3 | 3 | 0 | 0 | **100%** |
| **TOTAL** | **20** | **20** | **0** | **0** | **100%** |

> 📌 **Calibrated Threshold:** `0.50`  
> In-scope queries yield similarity scores between `0.65` and `0.88`, while out-of-scope queries consistently score below `0.35`. Setting the similarity threshold to 0.50 guarantees zero hallucinated answers while preserving accuracy on valid requests.

---

## ⚙️ Quick Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ammar8760/PolicyBot-InnoViast.git](https://github.com/Ammar8760/PolicyBot-InnoViast.git)
   cd PolicyBot-InnoViast