import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ==============================
# Load Chatbot Components
# ==============================

@st.cache_resource
def load_rag_components():
    # Load your dataset
    df = pd.read_csv("chatbot_chunks_final.csv")
    st.write(df.head())  # Debugging: Show first few rows to check if data is loaded correctly

    # Load precomputed embeddings
    embeddings = np.load("chatbot_embeddings.npy")
    st.write(embeddings.shape)  # Debugging: Check the shape of the embeddings

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    llm = pipeline("text2text-generation", model="google/flan-t5-large")

    return df, embeddings, embedder, llm


# ==============================
# Retrieve Context from Data
# ==============================

def retrieve_context(query, embedder, doc_embeddings, docs, top_k=5):
    # Encode the query
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {}
    
    # Calculate the similarity scores
    for idx, emb in enumerate(doc_embeddings):
        score = util.pytorch_cos_sim(query_emb, emb).item()
        scores[idx] = score

    # Sort by descending similarity and get the top K
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Debugging: Check if top docs were found
    st.write(f"Top docs: {top_docs}")

    try:
        context = "\n".join([docs['chunk'].iloc[idx] for idx, _ in top_docs])
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return "Error retrieving context."
    
    return context


# ==============================
# Query FLAN-T5 for Text Generation
# ==============================

def query_llm(query, context, llm):
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    output = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].strip()


# ==============================
# Chatbot Function (RAG + FLAN-T5)
# ==============================

def chatbot_page():
    st.subheader("🤖 Chatbot (Data Q&A)")

    user_input = st.text_input("Ask your question:")

    if user_input:
        with st.spinner("Thinking..."):
            # Load all required components
            df, embeddings, embedder, llm = load_rag_components()

            context = retrieve_context(user_input, embedder, embeddings, df)  # Retrieve relevant context
            response = query_llm(user_input, context, llm)  # Generate response from LLM
            
            st.write("Retrieved Context:")
            st.write(context)
            
            st.write("Answer:")
            st.write(response)


# ==============================
# Other Pages (Prediction, Graphs)
# ==============================

def predictor_page():
    st.subheader("🧠 Client Prediction")

    # Load prediction model (XGBoost model)
    model = joblib.load("XGB_model.jlib")

    # Define input fields for user to interact with
    # [Input form for client data here... e.g., age, dependents, etc.]

    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("Client is likely to return")
    else:
        st.warning("Client may not return")
    
    st.info(f"Probability of returning: {prob[1]*100:.2f}%")
    st.info(f"Probability of not returning: {prob[0]*100:.2f}%")


def graphs_page():
    st.subheader("📊 Feature Analysis Graphs")
    st.image("Graphs/flipupdate.png", caption="Feature Importance", use_container_width=True)
    st.image("Graphs/waterfall.png", caption="SNAP Waterfall Plot", use_container_width=True)


# ==============================
# Main Application
# ==============================

def main():
    st.title("Client Retention Predictor")

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Feature Analysis Graphs", "Chatbot (Data Q&A)"])

    if page == "Prediction":
        predictor_page()
    elif page == "Feature Analysis Graphs":
        graphs_page()
    elif page == "Chatbot (Data Q&A)":
        chatbot_page()


# Run the main function
if __name__ == "__main__":
    main()
