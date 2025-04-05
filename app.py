import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# ========== Load XGBoost Model ==========
model = joblib.load("XGB_model.jlib")

# ========== Load and Embed Chunks from finalfile.csv ==========
@st.cache_resource
def load_rag_components():
    df = pd.read_csv("finalfile.csv")

    # Build a natural language chunk from each row
    def row_to_chunk(row):
        return (
            f"A {int(row['age'])}-year-old {row['sex_new']} with {int(row['dependents_qty'])} dependents "
            f"picked up in {row['Month']} during {row['Season']}. "
            f"Distance: {row['distance_km']} km. Contact: {row['contact_method']} with {row['num_of_contact_methods']} method(s). "
            f"Language: {row['preferred_languages']}. Status: {row['status']}. Household: {row['household']}."
        )

    df['chunk'] = df.apply(row_to_chunk, axis=1)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = {
        idx: embedder.encode(text, convert_to_tensor=True)
        for idx, text in df['chunk'].items()
    }

    llm = pipeline("text2text-generation", model="google/flan-t5-large")
    return df, embedder, doc_embeddings, llm

# ========== RAG Chatbot Logic ==========
def retrieve_context(query, embedder, doc_embeddings, documents, top_k=2):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {
        idx: util.pytorch_cos_sim(query_embedding, emb).item()
        for idx, emb in doc_embeddings.items()
    }
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return "\n\n".join(documents['chunk'][idx] for idx, _ in top_docs)

def query_llm(query, context, llm):
    prompt = (
        "You have some background info and client summaries below.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\nAnswer:"
    )
    output = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return output[0]["generated_text"].replace(prompt, "").strip()

# ========== RAG Chatbot Page ==========
def chatbot_page_rag():
    st.title("🤖 RAG-Powered Chatbot (from CSV)")
    df, embedder, doc_embeddings, llm = load_rag_components()

    query = st.text_input("💬 Ask your question:")
    if st.button("Get Answer") and query:
        try:
            context = retrieve_context(query, embedder, doc_embeddings, df)
            answer = query_llm(query, context, llm)

            st.markdown("### 📄 Retrieved Context:")
            st.info(context)
            st.markdown("### 💬 Chatbot Answer:")
            st.success(answer)
        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")

# ========== App Navigation ==========
st.sidebar.title("Client Retention App")
page = st.sidebar.radio("Select a Page", [
    "Chatbot (RAG)"
])

if page == "Chatbot (RAG)":
    chatbot_page_rag()
