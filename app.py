import streamlit as st
import fitz
from utils import extract_title, chunk_text, build_index, generate_answer

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF Chatbot (RAG + Local LLM)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        title = extract_title(doc)

        all_text = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                all_text.append(text)

        chunks = []
        for t in all_text:
            chunks.extend(chunk_text(t))

        index, _ = build_index(chunks)

    st.success(f"Indexed PDF: **{title}**")

    query = st.text_input("Ask a question")

    if query:
        with st.spinner("Thinking..."):
            answer, sources = generate_answer(query, title, chunks, index)

        st.subheader("🧠 Answer")
        st.write(answer)

        if sources:
            st.subheader("📄 Sources")
            for s in sources:
                st.markdown(f"- {s[:300]}...")
