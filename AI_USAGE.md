# 🤖 AI Usage & Collaboration Report

This document outlines the collaborative engineering and design process between the developer and **Gemini** in building **PolicyBot (Nexora HR & Policy Assistant)**.

---

## 🤝 AI Collaboration Overview

AI was utilized as an advanced pairing partner to accelerate RAG pipeline design, engineer a dual vector database backend abstraction, fine-tune similarity score thresholds, build custom system prompts, and configure a secure, enterprise-grade deployment setup.

---

## ⚡ Feature Implementation & Refinement History

### 1. Secure Dual-Backend Architecture (ChromaDB & Pinecone)
* **Objective**: Enable high-speed, zero-cost offline local development while ensuring seamless, persistent vector indexing on ephemeral cloud platforms (Streamlit Community Cloud).
* **Collaboration & Solution**: Implemented a dynamic vector store manager that selects the database implementation based on environment variables (`VECTOR_BACKEND=pinecone` vs `VECTOR_BACKEND=chroma`).
* **Result**: Zero-configuration switching between offline local testing and live production deployment without changing the core query logic.

### 2. Pre-LLM Deterministic Score Threshold Gate
* **Objective**: Completely eliminate LLM hallucinations on out-of-scope or irrelevant queries while reducing unnecessary Gemini API token consumption.
* **Collaboration & Solution**: Intercepted retrieved vector hits post-similarity calculation and evaluated the top-1 score against a calibrated cutoff score (`0.50`). If `top_score < RELEVANCE_THRESHOLD`, the system immediately triggers a guided fallback instead of calling the LLM.
* **Result**: A deterministic boundary that rejects out-of-scope queries (e.g., general trivia or unlisted company policies) prior to calling the LLM.

### 3. Source Citation & Context Attribution Engine
* **Objective**: Ensure complete transparency by presenting exact document filenames, chunk numbers, similarity scores, and raw underlying policy text to end-users.
* **Collaboration & Solution**: Engineered a structured context builder that appends metadata directly into the system prompt and UI citation expanders (including source document name, chunk index, and calculated similarity score).
* **Result**: Users receive fully grounded, verifiable answers accompanied by collapsible source cards for auditability.

### 4. Synthetic Knowledge Base Validation & Consistency
* **Objective**: Create a robust, multi-document policy dataset ("Nexora Technologies") to stress-test complex multi-hop retrieval queries.
* **Collaboration & Solution**: Developed 5 synthetic policy manuals covering distinct operational domains:
  * **Leave & Attendance Policy**: Vacation, sick leave, and remote work rules.
  * **IT & Security Guidelines**: Hardware allocation, password rotation, and VPN usage.
  * **Expense & Reimbursement**: Travel allowances, meal limits, and receipt requirements.
  * **Code of Conduct**: Professional behavior, conflict of interest, and reporting channels.
  * **Remote Work Policy**: Equipment maintenance and core working hours.
* **Result**: Enabled reliable testing for both direct single-doc queries and multi-doc cross-referencing (e.g., laptop replacement timelines across IT and Remote policies).

---

## 🛡️ Responsible AI Use & Manual Verification

* **Manual Code Reviews**: Every module (`chunker.py`, `retriever.py`, `app.py`) and vector distance conversion formula was manually written, debugged, and validated locally.
* **Empirical Threshold Calibration**: The `0.50` similarity score cutoff was manually determined by executing 20 benchmark test queries and analyzing score distributions for in-scope vs. out-of-scope requests.
* **Secret Hygiene**: Verified that sensitive credentials (`GOOGLE_API_KEY`, `PINECONE_API_KEY`) are managed exclusively through `.env` and `st.secrets`—never committed to public GitHub repositories.