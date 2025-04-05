import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ===========================
# Load Chatbot Components
# ===========================
@st.cache_resource
def load_rag_components():
    # Load chunks and embeddings
    df = pd.read_csv("chatbot_chunks_final.csv")
    embeddings = np.load("chatbot_embeddings.npy")

    # Load Sentence Transformer and embed data
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = [torch.tensor(embeddings[i]) for i in range(len(df))]

    # Load FLAN-T5 generation pipeline
    llm = pipeline("text2text-generation", model="google/flan-t5-large")

    return df, embedder, doc_embeddings, llm

# ===========================
# Retrieve Context Function
# ===========================
def retrieve_context(query, embedder, doc_embeddings, docs, top_k=5):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {}

    for idx, emb in enumerate(doc_embeddings):
        score = util.pytorch_cos_sim(query_emb, emb).item()
        scores[idx] = score

    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return "\n\n".join([f"- {docs['chunk'].iloc[idx]}" for idx, _ in top_docs])

# ===========================
# LLM Query Function
# ===========================
def query_llm(query, context, llm):
    prompt = (
        "You are a helpful assistant summarizing client trends from a dataset.\n"
        "The following context contains summarized records.\n"
        "Use this context to answer the user's question as clearly and completely as possible.\n"
        "Do not make up information. Focus only on what's present in the data summaries.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {query}\n\n"
        "Answer:"
    )

    output = llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, "").strip()

# ===========================
# Chatbot Page
# ===========================
def chatbot_page():
    st.subheader("🤖 Chatbot (Data Q&A)")
    user_input = st.text_input("Ask your question here:")

    if user_input:
        with st.spinner("Generating answer..."):
            df, embedder, doc_embeddings, llm = load_rag_components()
            context = retrieve_context(user_input, embedder, doc_embeddings, df)
            answer = query_llm(user_input, context, llm)

        st.markdown("### 📄 Retrieved Context:")
        st.info(context)

        st.markdown("### 💬 Answer:")
        st.success(answer)

# ===========================
# Streamlit App Entry
# ===========================
def main():
    st.set_page_config(page_title="Client Data Chatbot", layout="wide")
    st.title("📊 Client Data Question Answering Chatbot")

    chatbot_page()

if __name__ == "__main__":
    main()
