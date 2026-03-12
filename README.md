# 📚 Multimodal PDF RAG Assistant

A production-ready **Retrieval Augmented Generation (RAG)** system for querying PDF documents using natural language. Built with a full hybrid retrieval pipeline — combining dense semantic search with sparse keyword search — for state-of-the-art retrieval accuracy.

---

## 🚀 Demo

Ask questions about any PDF and get accurate, context-grounded answers:

> *"How does U-Shape Transformer enhance underwater images?"*
> *"What PSNR and SSIM scores did the model achieve?"*
> *"What dataset was used for training?"*

**Evaluation results on test queries:**
- ✅ Hit Rate @5: **100%**
- ✅ MRR @5: **1.000**
- ✅ Cross-Encoder scores: **+7.8, +7.3, +4.8** (strongly positive)

---

## 🏗️ Architecture

```
PDF Input
    │
    ├── PyMuPDF text extraction
    └── Tesseract OCR (scanned pages)
         │
    Smart Chunking (1000 chars, 200 overlap)
         │
    ┌────┴────┐
    │         │
BGE-Large   BM25Okapi
Embeddings  Index
(Dense)     (Sparse)
    │         │
    └────┬────┘
    RRF Fusion (k=60)
         │
  Cross-Encoder Re-ranking
  (ms-marco-MiniLM-L-6-v2)
         │
    Ollama LLM
  (Llama 3.2 local)
         │
     Final Answer
```

**3-Stage Hybrid Retrieval Pipeline:**

| Stage | Component | Purpose |
|-------|-----------|---------|
| 1A | BGE-Large + Qdrant | Dense semantic retrieval |
| 1B | BM25Okapi | Sparse keyword retrieval |
| 2 | Reciprocal Rank Fusion | Combine both ranked lists |
| 3 | Cross-Encoder | Re-rank for final precision |

---

## ✨ Features

- 🔍 **Hybrid Retrieval** — Dense (BGE) + Sparse (BM25) fused via RRF
- 🧠 **BGE-Large Embeddings** — SOTA retrieval model (1024 dims)
- 🎯 **Cross-Encoder Re-ranking** — 30% better relevance vs vector-only
- 📄 **OCR Support** — Handles scanned PDFs via Tesseract
- 💾 **Qdrant Vector DB** — Persistent, production-ready storage
- 🤖 **Ollama LLM** — 100% local inference, zero API cost
- ✂️ **Smart Chunking** — Character-based with overlap (bug-fixed)
- 📊 **Retrieval Evaluation** — Hit Rate @K and MRR metrics
- 🖥️ **Streamlit UI** — Clean chat interface with source transparency

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | PyMuPDF (fitz) |
| OCR | Tesseract + pdf2image |
| Dense Embeddings | BAAI/bge-large-en-v1.5 |
| Sparse Retrieval | rank_bm25 (BM25Okapi) |
| Rank Fusion | Reciprocal Rank Fusion (RRF) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Vector Database | Qdrant (local persistent) |
| LLM | Ollama — Llama 3.2 1B (local) |
| UI | Streamlit |

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Tesseract OCR (optional, for scanned PDFs)

### Setup

```bash
# Clone the repository
git clone https://github.com/Yashraj0906/multimodal-pdf-rag-assistant.git
cd multimodal-pdf-rag-assistant

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama pull llama3.2:1b
```

### Requirements

```
pymupdf
pytesseract
pdf2image
sentence-transformers==3.3.1
qdrant-client==1.11.3
transformers==4.46.2
torch
pillow
matplotlib
tqdm
tf-keras
python-dotenv
requests
rank_bm25
streamlit
```

---

## 🚀 Running the App

```bash
# Start Ollama in a separate terminal
ollama serve

# Run the Streamlit app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Steps:**
1. Upload any PDF using the sidebar
2. Click **Process PDF** (takes ~40s for BGE-Large embeddings)
3. Ask questions in the chat box
4. Toggle **Use LLM Answer** for generated responses vs raw passages

---

## 📓 Notebook

The Jupyter notebook `Multimodal_Research_Paper_RAG_Assistant_ENHANCED.ipynb` walks through the complete pipeline step by step:

| Step | Description |
|------|-------------|
| 1-2 | Install packages + imports |
| 3-4 | Configuration + model loading |
| 5-7 | PDF text + image extraction |
| 8 | Smart chunking + BM25 index |
| 9-10 | BGE embeddings + Qdrant indexing |
| 11 | Hybrid search function (3-stage) |
| 12-13 | Test queries + Ollama answer generation |
| 14 | Interactive Q&A interface |
| 15 | Dense vs Hybrid comparison |
| 16 | Retrieval evaluation (Hit Rate + MRR) |

---

## 📊 Performance

Tested on a 14-page IEEE research paper (underwater image enhancement):

| Metric | Value |
|--------|-------|
| Chunks indexed | 95 |
| Embedding dimensions | 1024 |
| Avg chunk size | 889 chars |
| Indexing time | ~40s (BGE-Large, CPU) |
| Query latency | <2s (retrieval + re-ranking) |
| LLM answer time | ~10s (Llama 3.2 1B, CPU) |
| Hit Rate @5 | 100% |
| MRR @5 | 1.000 |

---

## 🔍 Why Hybrid Retrieval?

| Method | Strength | Weakness |
|--------|----------|----------|
| Dense only (BGE) | Semantic meaning, handles synonyms | Misses exact keyword matches |
| Sparse only (BM25) | Exact term matching, rare words | Misses paraphrases, context |
| **Hybrid (RRF)** | **Best of both worlds** | Slightly more complex |

**Reciprocal Rank Fusion formula:** `score = 1/(k + rank)` where k=60

A chunk ranked #1 by dense AND #3 by BM25 scores higher than a chunk ranked #1 by only one method — rewarding consistency across retrieval signals.

---

## ⚠️ Known Limitations

- Images extracted but not used for retrieval (true multimodal would use CLIP/LLaVA)
- Tesseract OCR quality degrades on low-resolution scans
- Llama 3.2 1B is a small model — complex questions may produce imprecise answers
- Local Qdrant allows only one process at a time (use Qdrant server for concurrent access)
- Cross-encoder trained on MS MARCO (web Q&A) — domain mismatch on technical papers

---

## 🗺️ Roadmap

- [ ] CLIP image embeddings for true multimodal retrieval
- [ ] Streaming LLM responses in Streamlit
- [ ] Multi-document search with per-document filtering
- [ ] RAGAS evaluation framework integration
- [ ] Docker containerization
- [ ] Qdrant Cloud support for multi-user deployment

---

## 📁 Project Structure

```
multimodal-pdf-rag-assistant/
├── app.py                          # Streamlit web application
├── Multimodal_Research_Paper_      # Jupyter notebook (full pipeline)
│   RAG_Assistant_ENHANCED.ipynb
├── requirements.txt                # Python dependencies
├── README.md
├── .gitignore
├── data/
│   └── qdrant_storage/             # Qdrant persistent storage (auto-created)
└── images/                         # Extracted PDF images (auto-created)
```

---

## 👤 Author

**Yashraj Kumar**
B.Tech ECE — IIIT Nagpur
[GitHub](https://github.com/Yashraj0906) · [LinkedIn](https://linkedin.com/in/yashraj-kumar-3b0673287)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
