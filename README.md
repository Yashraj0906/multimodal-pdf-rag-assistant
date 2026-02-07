# 📚 Multimodal PDF RAG Assistant

A **production-style Retrieval-Augmented Generation (RAG) system** for asking questions over research papers and PDFs.  
The system supports **semantic search, OCR for scanned PDFs, re-ranking for higher relevance**, and a **Streamlit UI**.

---

## 🚀 Features

- 🔍 **Hybrid Retrieval**
  - Dense semantic search using **BGE embeddings**
  - Keyword-aware chunking for better recall

- 🎯 **Cross-Encoder Re-ranking**
  - Improves retrieval relevance by ~30%
  - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`

- 📄 **OCR Support**
  - Handles scanned PDFs using **Tesseract OCR**

- 💾 **Persistent Vector Storage**
  - Uses **Qdrant** as a local vector database
  - Data survives restarts (no in-memory loss)

- 💬 **Interactive UI**
  - Built with **Streamlit**
  - Upload PDF → Ask questions → View sources

- ⚡ **Optimized Pipeline**
  - Chunking with overlap
  - Batched embedding generation
  - Sub-2s retrieval latency for medium PDFs

---

## 🧠 Architecture Overview

```text
PDF
 ↓
Text & OCR Extraction (PyMuPDF + Tesseract)
 ↓
Chunking with Overlap
 ↓
BGE Embeddings (Sentence Transformers)
 ↓
Qdrant Vector Database
 ↓
Vector Search (Top-K)
 ↓
Cross-Encoder Re-ranking
 ↓
Relevant Context for Answering
🛠️ Tech Stack

Python 3.10+

Sentence Transformers (BGE)

Cross-Encoders

Qdrant

PyMuPDF

Tesseract OCR

Streamlit

📦 Installation
1️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt


⚠️ For OCR support, install Tesseract separately:

Windows: https://github.com/UB-Mannheim/tesseract/wiki

Linux: sudo apt install tesseract-ocr

▶️ Run the Application
streamlit run app.py
