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
    df = pd.read_csv("chatbot_chunks.csv")
    embeddings = np.load("chatbot_embeddings.npy")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = [torch.tensor(embeddings[i]) for i in range(len(df))]
    llm = pipeline("text2text-generation", model="google/flan-t5-large")

    return df, embedder, doc_embeddings, llm

def retrieve_context(query, embedder, doc_embeddings, docs, top_k=5):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {}

    for idx, emb in enumerate(doc_embeddings):
        score = util.pytorch_cos_sim(query_emb, emb).item()
        scores[idx] = score

    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return "\n\n".join([f"- {docs['chunk'].iloc[idx]}" for idx, _ in top_docs])

def query_llm(query, context, llm):
    prompt = (
        "You are a helpful assistant summarizing client trends from a dataset.\n"
        "The following context contains detailed summaries from real client records.\n"
        "Use the information to answer the user’s question as fully and clearly as possible.\n"
        "If the user asks a general question (like 'what is this data about?'), provide a full summary.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {query}\n\n"
        "Answer:"
    )

    output = llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, "").strip()

# ===========================
# Chatbot UI Page
# ===========================
def chatbot_page():
    st.subheader("🤖 Chatbot (Data Q&A)")
    user_input = st.text_input("Ask your question:")

    if user_input:
        with st.spinner("Analyzing your question..."):
            df, embedder, doc_embeddings, llm = load_rag_components()
            context = retrieve_context(user_input, embedder, doc_embeddings, df)
            answer = query_llm(user_input, context, llm)

        st.markdown("### 📄 Retrieved Context:")
        st.info(context)

        st.markdown("### 💬 Answer:")
        st.success(answer)

# ===========================
# Predictor Page (Optional)
# ===========================
def predictor_page():
    st.subheader("📈 Client Return Predictor")

    age = st.slider("Age", 18, 100, 30)
    dependents_qty = st.number_input("Dependents", min_value=0, max_value=10, value=1)
    distance_km = st.number_input("Distance to Location (km)", min_value=0.0, max_value=100.0, value=5.0)
    num_of_contact_methods = st.number_input("Number of Contact Methods", min_value=1, max_value=5, value=2)
    household = st.selectbox("Has a household", ["Yes", "No"])
    sex = st.selectbox("Sex", ["Male", "Female"])
    status = st.selectbox("Status", ["Active", "Closed", "Flagged", "Outreach"])
    latest_lang_english = st.selectbox("Latest language is English", ["Yes", "No"])
    season = st.selectbox("Season of Pickup", ["Fall", "Spring", "Summer", "Winter"])
    month = st.selectbox("Month of Pickup", [
        "April", "August", "December", "February", "January", "July", "June",
        "March", "May", "November", "October", "September"
    ])

    submitted = st.button("Predict")

    if submitted:
        d = {
            'age': [age],
            'dependents_qty': [dependents_qty],
            'distance_km': [distance_km],
            'num_of_contact_methods': [num_of_contact_methods]
        }

        df_input = pd.DataFrame(d)
        df_input["household_Yes"] = 1 if household == "Yes" else 0
        df_input["sex_new_Male"] = 1 if sex == "Male" else 0
        df_input[f"status_{status}"] = 1
        df_input["latest_language_is_english_Yes"] = 1 if latest_lang_english == "Yes" else 0
        df_input[f"Season_{season}"] = 1
        df_input[f"Month_{month}"] = 1

        model = joblib.load("XGB_model.jlib")
        prediction = model.predict(df_input)[0]
        probs = model.predict_proba(df_input)[0]

        st.subheader("📊 Prediction Result:")
        if prediction == 1:
            st.success("✅ Client is likely to return")
        else:
            st.warning("⚠️ Client may not return")

        st.info(f"📈 Probability of returning: {probs[1]*100:.2f}%")
        st.info(f"📉 Probability of not returning: {probs[0]*100:.2f}%")

# ===========================
# Graphs Page (Optional)
# ===========================
def graphs_page():
    st.subheader("📊 Feature Analysis Graphs")
    st.image("Graphs/fipupdate.png", caption="Feature Importance", use_container_width=True)
    st.image("Graphs/waterfall.png", caption="SHAP Waterfall Plot", use_container_width=True)

# ===========================
# Main Navigation
# ===========================
def main():
    st.set_page_config(page_title="Client Data Q&A", layout="wide")
    st.title("📌 Client Retention Analysis")

    page = st.sidebar.radio("Choose a section:", [
        "Chatbot (Data Q&A)",
        "Client Return Predictor",
        "Feature Analysis Graphs"
    ])

    if page == "Chatbot (Data Q&A)":
        chatbot_page()
    elif page == "Client Return Predictor":
        predictor_page()
    elif page == "Feature Analysis Graphs":
        graphs_page()

if __name__ == "__main__":
    main()
