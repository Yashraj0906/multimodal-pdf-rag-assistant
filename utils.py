import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Load model once
text_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def extract_title(doc):
    first_page = doc[0].get_text()
    return first_page.split("\n")[0].strip()

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

def build_index(chunks):
    embeddings = text_model.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve(query, chunks, index, k=3):
    q_emb = text_model.encode([query])
    _, idx = index.search(q_emb, k)
    return [chunks[i] for i in idx[0]]

def build_prompt(query, contexts):
    context = "\n\n".join(contexts)
    return f"""
You are an AI assistant.
Answer ONLY using the context below.
If not found, say "Not found in the document."

Context:
{context}

Question:
{query}

Answer:
"""

def generate_answer(query, title, chunks, index):
    q = query.lower()

    # 🔹 Query routing for metadata
    if "title" in q or "name of the paper" in q:
        return title, []

    contexts = retrieve(query, chunks, index)
    prompt = build_prompt(query, contexts)

    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"], contexts
