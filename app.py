"""
PDF RAG Chatbot (Hybrid Search)
- BGE Embeddings
- Qdrant Vector DB
- Cross-Encoder Re-ranking
- Streamlit UI
"""

import streamlit as st
import fitz  # PyMuPDF
import os
from uuid import uuid4
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📚 PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    qdrant = QdrantClient(path="./qdrant_data")
    return embedding_model, cross_encoder, qdrant

# ---------------- PDF PROCESSING ----------------
def process_pdf(pdf_file, embedding_model, qdrant):
    pdf_path = f"temp_{pdf_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())

    doc = fitz.open(pdf_path)
    text_chunks = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            text_chunks.append({
                "text": text,
                "page": page_num + 1
            })

    doc.close()
    os.remove(pdf_path)

    if not text_chunks:
        return 0

    # Embeddings
    instruction = "Represent this document for retrieval: "
    texts = [instruction + c["text"] for c in text_chunks]
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Recreate collection
    collection_name = "pdf_docs"
    try:
        qdrant.delete_collection(collection_name)
    except:
        pass

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embeddings.shape[1],
            distance=Distance.COSINE
        )
    )

    points = []
    for chunk, emb in zip(text_chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"]
                }
            )
        )

    qdrant.upsert(collection_name=collection_name, points=points)
    return len(text_chunks)

# ---------------- SEARCH FUNCTION (FIXED) ----------------
def search_query(query, embedding_model, cross_encoder, qdrant, top_k=5):
    instruction = "Represent this query for retrieval: "
    query_emb = embedding_model.encode(
        instruction + query,
        normalize_embeddings=True
    )

    results = qdrant.search(
        collection_name="pdf_docs",
        query_vector=query_emb.tolist(),
        limit=top_k * 2
    )

    documents = [
        {
            "text": r.payload["text"],
            "page": r.payload["page"],
            "score": r.score
        }
        for r in results
    ]

    # 🚨 SAFETY CHECK (CRITICAL FIX)
    if not documents:
        return []

    # Prepare re-ranking pairs
    pairs = [[query, d["text"]] for d in documents]

    if not pairs:
        return documents[:top_k]

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["relevance_score"] = float(score)

    documents = sorted(
        documents,
        key=lambda x: x["relevance_score"],
        reverse=True
    )[:top_k]

    return documents

# ---------------- UI ----------------
st.title("📚 PDF Question Answering (Hybrid RAG)")

st.markdown("""
**Features**
- 🔍 Hybrid Search (Vector + Re-ranking)
- 🧠 BGE Embeddings
- 💾 Qdrant Vector DB
- ⚡ Robust & Production-Safe
""")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Setup")

    with st.spinner("Loading models..."):
        embedding_model, cross_encoder, qdrant = load_models()
    st.success("Models loaded")

    st.divider()

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            chunks = process_pdf(uploaded_file, embedding_model, qdrant)
            st.session_state.processed = chunks > 0
        if chunks > 0:
            st.success(f"Processed {chunks} chunks")
        else:
            st.error("No text found in PDF")

    st.divider()
    top_k = st.slider("Top-K Results", 3, 10, 5)

# ---------------- CHAT ----------------
st.header("💬 Chat")

if not st.session_state.processed:
    st.info("Upload and process a PDF first.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                results = search_query(
                    prompt,
                    embedding_model,
                    cross_encoder,
                    qdrant,
                    top_k
                )

                if not results:
                    answer = "No relevant information found in the document."
                    st.warning(answer)
                else:
                    context = "\n\n".join(
                        f"**Page {r['page']}**:\n{r['text'][:300]}..."
                        for r in results
                    )

                    answer = f"""
Here are the most relevant passages:

{context}

**(Retrieved using hybrid search with re-ranking)**
"""
                    st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

st.divider()
st.caption("Built with BGE, Qdrant, Cross-Encoders, Streamlit | 2026-ready RAG system")
