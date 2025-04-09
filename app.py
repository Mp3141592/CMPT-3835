# chatbot_app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load data
@st.cache_data(show_spinner=False)
def load_chunks():
    df = pd.read_csv("chatbot_chunks_final.csv")
    return {f"chunk_{i}": row["chunk"] for i, row in df.iterrows()}

# Load models
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-large")
    return embedder, generator

# Embed all documents
def embed_documents(embedder, docs):
    return {
        doc_id: embedder.encode(text, convert_to_tensor=True)
        for doc_id, text in docs.items()
    }

# Retrieve relevant context
def retrieve_context(query, doc_embeddings, docs, embedder, top_k=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {
        doc_id: util.pytorch_cos_sim(query_embedding, emb).item()
        for doc_id, emb in doc_embeddings.items()
    }
    top_doc_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return "\n\n".join(docs[doc_id] for doc_id in top_doc_ids)

# Query FLAN-T5
def generate_answer(query, context, generator):
    prompt = (
        "You are a helpful assistant for IFSSA. Use the context below to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer:"
    )
    output = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, "").strip()

# Streamlit UI
st.set_page_config(page_title="IFSSA Chatbot", page_icon="🤖")
st.title("🤖 IFSSA Chatbot")
st.write("Ask a question about IFSSA, food hampers, or client services.")

# Load models and chunks
docs = load_chunks()
embedder, generator = load_models()
doc_embeddings = embed_documents(embedder, docs)

# User input
user_query = st.text_input("What would you like to ask?")

if user_query:
    with st.spinner("Generating answer..."):
        context = retrieve_context(user_query, doc_embeddings, docs, embedder)
        answer = generate_answer(user_query, context, generator)
        st.markdown("### 📌 Answer:")
        st.write(answer)
