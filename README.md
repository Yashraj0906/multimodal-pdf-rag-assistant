# 📄 Multimodal PDF RAG Chatbot

A **production-style multimodal Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload any PDF document and ask questions grounded strictly in the document content.

The system combines **semantic retrieval**, **metadata-aware query routing**, and a **local open-source LLM** to deliver accurate and explainable answers.

---

## 🚀 Features

- 📤 Upload and chat with **any PDF**
- 🔍 Dense semantic retrieval using **BGE embeddings**
- 🧠 Grounded answer generation using **local LLM (Ollama – Gemma)**
- 🧾 Explicit metadata handling (paper title routing)
- ⚡ Fast vector search with **FAISS**
- 🖥️ Interactive UI built with **Streamlit**
- 🔐 Fully offline – no OpenAI or paid APIs

---

## 🧠 Architecture Overview

PDF Upload
↓
Text Extraction (PyMuPDF)
↓
Chunking + Metadata
↓
BGE Embeddings
↓
FAISS Vector Index
↓
Query Routing
↓
Local LLM (Ollama)
↓
Answer + Source Chunks

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI
- **PyMuPDF** – PDF parsing
- **Sentence-Transformers (BGE)** – text embeddings
- **FAISS** – vector similarity search
- **Ollama (Gemma)** – local LLM inference
- **PyTorch**

---

## 📁 Project Structure

pdf-multimodal-rag-chatbot/
│── app.py # Streamlit application
│── utils.py # Core RAG logic
│── requirements.txt # Dependencies
│── paper.pdf # Sample PDF
│── images/ # Extracted images (optional)

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/pdf-multimodal-rag-chatbot.git
cd pdf-multimodal-rag-chatbot
2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Install & run Ollama

Download from: https://ollama.com

ollama run gemma3:1b

▶️ Run the App
streamlit run app.py
