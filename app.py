import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# Load Chatbot Components
@st.cache_resource
def load_rag_components():
    df = pd.read_csv("chatbot_chunks.csv")
    embeddings = np.load("chatbot_embeddings.npy")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return df, embeddings, embedder

# Retrieve context based on query
def retrieve_context(query, embedder, doc_embeddings, top_k=5):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {}

    for idx, emb in enumerate(doc_embeddings):
        score = util.pytorch_cos_sim(query_emb, emb).item()
        scores[idx] = score

    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    context = "\n".join([docs['chunk'].iloc[idx] for idx, _ in top_docs])
    return context

# Query FLAN-T5 for Text Generation
def query_llm(query, context, llm):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    output = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].strip()

# Chatbot Function (RAG + FLAN-T5)
def chatbot_page():
    st.subheader("🧠 Chatbot (Data Q&A)")
    user_input = st.text_input("Ask your question:")

    if user_input:
        with st.spinner("Thinking..."):
            df, embeddings, embedder = load_rag_components()
            llm = pipeline("text2text-generation", model="google/flan-t5-large")

            context = retrieve_context(user_input, embedder, embeddings)  # Retrieve relevant context
            answer = query_llm(user_input, context, llm)  # Generate answer

            st.write("Retrieved Context:")
            st.write(context)

            st.write("Answer:")
            st.write(answer)

# Model Prediction (Client Retention Prediction)
def predictor_page():
    st.subheader("🧠 Client Prediction")

    # Load the model (ensure XGB_model.jlib is in the same directory)
    model = joblib.load("XGB_model.jlib")

    # Define input fields for user to interact with (Make sure this matches your input features)
    household = st.selectbox("Does the client have a household?", ['Yes', 'No'])
    sex = st.selectbox("Sex of client", ['Male', 'Female'])
    dependents_qty = st.slider("Number of Dependents", min_value=0, max_value=10, value=3)
    distance_km = st.slider("Distance to Pickup (in km)", min_value=1, max_value=50, value=7)

    # Ensure input is properly processed
    df_input = pd.DataFrame({
        'household_yes': [1 if household == 'Yes' else 0],
        'sex_newMale': [1 if sex == 'Male' else 0],
        'dependents_qty': [dependents_qty],
        'distance': [distance_km],
    })

    # Debugging line: Check the input DataFrame
    st.write("Data input to model:")
    st.write(df_input)

    try:
        prediction = model.predict(df_input)[0]  # Get prediction
        prob = model.predict_proba(df_input)[0]  # Get probabilities

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success("Client is likely to return")
        else:
            st.warning("Client may not return")
        
        st.info(f"Probability of returning: {prob[1]*100:.2f}%")
        st.info(f"Probability of not returning: {prob[0]*100:.2f}%")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Graphs Page (Optional: Graphs and Analysis)
def graphs_page():
    st.subheader("📊 Feature Analysis Graphs")
    st.image("Graphs/fiupdate.png", caption="Feature Importance", use_container_width=True)
    st.image("Graphs/waterfall.png", caption="SNAP Waterfall Plot", use_container_width=True)

# Main Function
def main():
    st.title("Client Retention Predictor")

    # Add a selectbox for navigation
    page = st.sidebar.selectbox("Select Page", ["Client Prediction", "Chatbot (Data Q&A)", "Feature Analysis Graphs"])

    if page == "Client Prediction":
        predictor_page()
    elif page == "Chatbot (Data Q&A)":
        chatbot_page()
    elif page == "Feature Analysis Graphs":
        graphs_page()

if __name__ == "__main__":
    main()
