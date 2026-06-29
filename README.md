# 💼 Enterprise Knowledge Assistant (Advanced RAG Pipeline)

An enterprise-grade, high-performance Advanced Retrieval-Augmented Generation (RAG) system built with **LangChain (LCEL)**, **Groq Cloud Infrastructure**, and **Streamlit**. This assistant allows secure, multi-document semantic queries with real-time evaluation metrics, context-aware memory rewriting, and a localized security gateway.

---

## 🏗️ 1. Architecture Overview

This system utilizes a highly decoupled, production-oriented modular architecture divided into clean processing stages:

* **🚀 Ingestion Component (`ingest.py`):** Parses raw multi-format corporate records (`.pdf`), handles sliding-window document chunking, and maps structural page-level metadata variables.
* **📦 Vector Shard Index (`chroma_db`):** Embeds processed text fragments via local `sentence-transformers/all-MiniLM-L6-v2` neural models and indexes them inside a persistent, disk-backed Chroma vector store.
* **🧠 Contextual Query Rewriter:** An LCEL pipeline that analyzes user query arrays against active state memory logs to automatically synthesize standalone, context-complete search expressions.
* **🔍 Advanced Retrieval & Inference Engine:** Queries the index database nodes, passes context vectors securely to the **Llama-3.1-8b-instant** hardware matrix over Groq API pipelines, and formats citations under explicit anti-hallucination guardrails.

---

## ⚡ 2. Core Advanced System Features

* **🔐 Enterprise Security Gateway:** A built-in security perimeter requiring dynamic access tokens (`admin123`) to unlock core data assets.
* **🏎️ RAM Optimization Layer:** Wrapped using Streamlit `@st.cache_resource` decorators to load embedding vectors into host memory exactly once, bypassing heavy boot lags.
* **📊 RAGAS Analytics Integration:** A dedicated telemetry dashboard tracking system-level performance indices (**Faithfulness**, **Answer Relevance**, and **Context Precision**).
* **📄 Auditable Document Citation Trails:** Fully interactive UI expanders revealing the file source names, page counts, and specific context fragments used for generating responses.

---

## 🛠️ 3. Setup & Installation Instructions

Follow these instructions to configure and deploy the active pipeline environment locally:

### Step 1: Install Dependencies
Ensure you are using Python 3.10+ and run the installation script:
```bash
pip install -r requirements.txt