"""
PDF RAG Chatbot — Full Hybrid Search
Pipeline: BGE Dense + BM25 Sparse → RRF Fusion → Cross-Encoder → Ollama LLM
"""

import streamlit as st
import fitz  # PyMuPDF
import os
import numpy as np
import requests

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

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
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "chunked_texts" not in st.session_state:
    st.session_state.chunked_texts = []
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    qdrant = QdrantClient(path="./qdrant_data")
    return embedding_model, cross_encoder, qdrant

# ---------------- CHUNKING ----------------
def chunk_text_with_overlap(text, chunk_size=1000, overlap=200):
    """Character-based chunking with overlap."""
    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + sentence + ". "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# ---------------- PDF PROCESSING ----------------
def process_pdf(pdf_file, embedding_model, qdrant):
    pdf_path = f"temp_{pdf_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())

    doc = fitz.open(pdf_path)
    all_chunks = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            page_chunks = chunk_text_with_overlap(text)
            for chunk in page_chunks:
                all_chunks.append({
                    "text": chunk,
                    "page": page_num + 1,
                    "chunk_id": len(all_chunks)
                })

    doc.close()
    os.remove(pdf_path)

    if not all_chunks:
        return 0, [], None

    # Generate BGE embeddings
    instruction = "Represent this document for retrieval: "
    texts = [instruction + c["text"] for c in all_chunks]

    with st.spinner(f"Generating embeddings for {len(all_chunks)} chunks..."):
        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )

    # Build BM25 index for sparse retrieval
    bm25_corpus = [c["text"].lower().split() for c in all_chunks]
    bm25_index = BM25Okapi(bm25_corpus)

    # Recreate Qdrant collection clean
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

    # Integer IDs — reliable across all qdrant-client versions
    points = []
    for idx, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
        points.append(
            PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "chunk_id": idx
                }
            )
        )

    for i in range(0, len(points), 100):
        qdrant.upsert(collection_name=collection_name, points=points[i:i+100])

    return len(all_chunks), all_chunks, bm25_index

# ---------------- HYBRID SEARCH ----------------
def search_query(query, embedding_model, cross_encoder, qdrant, chunked_texts, bm25_index, top_k=5):
    """
    Full 3-stage hybrid retrieval:
    Stage 1A: Dense  — BGE embeddings + Qdrant cosine search
    Stage 1B: Sparse — BM25 keyword search
    Stage 2:  RRF    — Reciprocal Rank Fusion combines both lists
    Stage 3:  CE     — Cross-Encoder re-ranks for final precision
    
    CE scores are raw logits — higher = more relevant, can be negative (normal)
    """

    # Stage 1A: Dense retrieval
    instruction = "Represent this query for retrieval: "
    query_emb = embedding_model.encode(instruction + query, normalize_embeddings=True)

    dense_results = qdrant.search(
        collection_name="pdf_docs",
        query_vector=query_emb.tolist(),
        limit=top_k * 3
    )

    dense_ranks = {}
    for rank, result in enumerate(dense_results):
        chunk_id = result.payload["chunk_id"]
        dense_ranks[chunk_id] = rank + 1

    # Stage 1B: BM25 sparse retrieval
    query_tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]

    bm25_ranks = {}
    for rank, idx in enumerate(top_bm25_indices):
        bm25_ranks[int(idx)] = rank + 1

    # Stage 2: Reciprocal Rank Fusion (k=60 is standard)
    k_rrf = 60
    all_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())

    rrf_scores = {}
    for cid in all_ids:
        rrf_scores[cid] = (
            1 / (k_rrf + dense_ranks.get(cid, 1000)) +
            1 / (k_rrf + bm25_ranks.get(cid, 1000))
        )

    top_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k * 2]

    documents = []
    for cid in top_ids:
        if cid < len(chunked_texts):
            chunk = chunked_texts[cid]
            documents.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "rrf_score": rrf_scores[cid],
                "dense_rank": dense_ranks.get(cid, "—"),
                "sparse_rank": bm25_ranks.get(cid, "—")
            })

    if not documents:
        return []

    # Stage 3: Cross-encoder re-ranking
    pairs = [[query, d["text"]] for d in documents]
    ce_scores = cross_encoder.predict(pairs)

    for doc, score in zip(documents, ce_scores):
        doc["relevance_score"] = float(score)

    documents = sorted(documents, key=lambda x: x["relevance_score"], reverse=True)[:top_k]
    return documents

# ---------------- OLLAMA ----------------
def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False

def generate_answer(query, context_docs, model="llama3.2:1b"):
    context = "\n\n".join([
        f"[Page {doc['page']}]\n{doc['text']}"
        for doc in context_docs[:5]
    ])
    prompt = f"""Based on the following context from a document, answer the question clearly and concisely.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        return response.json().get("response", "Could not generate answer.")
    except Exception as e:
        return f"Ollama error: {e}"

# ---------------- UI ----------------
st.title("📚 PDF Question Answering — Hybrid RAG")
st.markdown("**Pipeline:** BGE Dense + BM25 Sparse → RRF Fusion → Cross-Encoder → Ollama LLM")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Setup")

    with st.spinner("Loading models..."):
        embedding_model, cross_encoder, qdrant = load_models()
    st.success("✅ Models loaded")

    ollama_running = check_ollama()
    if ollama_running:
        st.success("✅ Ollama running")
    else:
        st.warning("⚠️ Ollama not running\nRun: `ollama serve`")

    st.divider()

    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

    if uploaded_file:
        if st.button("🚀 Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                chunks, chunked_texts, bm25_index = process_pdf(
                    uploaded_file, embedding_model, qdrant
                )
                st.session_state.processed = chunks > 0
                st.session_state.chunk_count = chunks
                st.session_state.chunked_texts = chunked_texts
                st.session_state.bm25_index = bm25_index
                st.session_state.messages = []

            if chunks > 0:
                st.success(f"✅ {chunks} chunks indexed")
                st.info("🔍 Dense + BM25 indexes ready")
            else:
                st.error("❌ No text found in PDF")

    if st.session_state.processed:
        st.info(f"📊 {st.session_state.chunk_count} chunks ready")

    st.divider()
    top_k = st.slider("Top-K Results", 3, 10, 5)
    use_llm = st.toggle("Use LLM Answer", value=ollama_running, disabled=not ollama_running)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- CHAT ----------------
st.header("💬 Chat")

if not st.session_state.processed:
    st.info("👈 Upload and process a PDF from the sidebar first.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Hybrid searching..."):
                results = search_query(
                    prompt,
                    embedding_model,
                    cross_encoder,
                    qdrant,
                    st.session_state.chunked_texts,
                    st.session_state.bm25_index,
                    top_k
                )

            if not results:
                answer = "❌ No relevant information found."
                st.warning(answer)
            else:
                if use_llm and ollama_running:
                    with st.spinner("Generating answer with Ollama..."):
                        llm_answer = generate_answer(prompt, results)

                    answer = f"**💡 Answer:**\n\n{llm_answer}"
                    st.markdown(answer)

                    with st.expander(f"📚 View {len(results)} source passages"):
                        for i, r in enumerate(results, 1):
                            ce = r.get('relevance_score', 0)
                            st.markdown(
                                f"**[{i}] Page {r['page']}** | "
                                f"CE: `{ce:.3f}` | "
                                f"Dense: `{r.get('dense_rank','—')}` | "
                                f"BM25: `{r.get('sparse_rank','—')}`"
                            )
                            st.markdown(r['text'][:400] + "...")
                            st.divider()
                else:
                    passages = "\n\n".join(
                        f"**[{i}] Page {r['page']}** (CE: `{r.get('relevance_score',0):.3f}` | Dense: `{r.get('dense_rank','—')}` | BM25: `{r.get('sparse_rank','—')}`):\n{r['text'][:300]}..."
                        for i, r in enumerate(results, 1)
                    )
                    answer = f"**📚 Most relevant passages:**\n\n{passages}"
                    st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("BGE-Large · BM25 · RRF · Cross-Encoder · Ollama · Qdrant | Hybrid RAG 2026")
