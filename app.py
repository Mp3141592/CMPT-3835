import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# Load ML model
model = joblib.load("XGB_model.jlib")

# ================== Load and Embed CSV for Chatbot ==================
@st.cache_data
def load_csv_chunks():
    df = pd.read_csv("finalfile.csv")

    def row_to_chunk(row):
        return (
            f"A {int(row['age'])}-year-old {row['sex_new']} with {int(row['dependents_qty'])} dependents "
            f"picked up in {row['Month']} during {row['Season']}. "
            f"Distance: {row['distance_km']} km. Contact: {row['contact_method']} with {row['num_of_contact_methods']} method(s). "
            f"Language: {row['preferred_languages']}. Status: {row['status']}. Household: {row['household']}."
        )

    df['chunk'] = df.apply(row_to_chunk, axis=1)
    return df

@st.cache_resource
def load_embeddings_model(df):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = {
        idx: embedder.encode(text, convert_to_tensor=True)
        for idx, text in df['chunk'].items()
    }
    llm = pipeline("text2text-generation", model="google/flan-t5-large")
    return embedder, doc_embeddings, llm

# ================== Retrieval + Prompt ==================
def retrieve_context(query, embedder, doc_embeddings, documents, top_k=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {
        idx: util.pytorch_cos_sim(query_embedding, emb).item()
        for idx, emb in doc_embeddings.items()
    }
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return "\n".join(f"- {documents['chunk'][idx]}" for idx, _ in top_docs)

def query_llm(query, context, llm):
    prompt = (
        "You are a helpful assistant. Below are several summaries of client records.\n"
        "Summarize the information and answer the question clearly using your own words.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    output = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return output[0]["generated_text"].replace(prompt, "").strip()

# ================== Chatbot Page ==================
def chatbot_page_rag():
    st.title("🤖 RAG Chatbot (from finalfile.csv)")
    df = load_csv_chunks()
    embedder, doc_embeddings, llm = load_embeddings_model(df)

    query = st.text_input("💬 Ask your question:")
    if st.button("Get Answer") and query:
        try:
            context = retrieve_context(query, embedder, doc_embeddings, df)
            answer = query_llm(query, context, llm)
            st.markdown("### 📄 Retrieved Context:")
            st.info(context)
            st.markdown("### 💬 Answer:")
            st.success(answer)
        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")

# ================== Sidebar Navigation ==================
st.sidebar.title("Client Retention App")
page = st.sidebar.radio("Select a Page", [
    "Client Retention Predictor",
    "Feature Analysis Graphs",
    "Chatbot (RAG)"
])

# ================== Predictor Tab ==================
if page == "Client Retention Predictor":
    st.title("🔄 Client Retention Predictor")

    season = st.selectbox("Season of Pickup", ["Select a season", "Spring", "Summer", "Fall", "Winter"])
    season_months = {
        'Spring': ['March', 'April', 'May'],
        'Summer': ['June', 'July', 'August'],
        'Fall': ['September', 'October', 'November'],
        'Winter': ['December', 'January', 'February']
    }

    if season != "Select a season":
        month = st.selectbox("Month of Pickup", season_months[season])

        with st.form("prediction_form"):
            age = st.slider("Age", 18, 100, 35)
            dependents_qty = st.number_input("Dependents", 0, 12, 1)
            distance_km = st.number_input("Distance to Location (km)", 0.0, 210.0, 5.0)
            num_of_contact_methods = st.slider("Number of Contact Methods", 1, 5, 2)
            household = st.selectbox("Has a household", ['Yes', 'No'])
            sex = st.selectbox("Gender", ['Male', 'Female'])
            status = st.selectbox("Status", ['Active', 'Closed', 'Pending', 'Outreach', 'Flagged'])
            latest_lang_english = st.selectbox("Latest Language is English", ['Yes', 'No'])
            submitted = st.form_submit_button("Predict")

        if submitted:
            d = {
                'age': [age],
                'dependents_qty': [dependents_qty],
                'distance_km': [distance_km],
                'num_of_contact_methods': [num_of_contact_methods]
            }

            df_input = pd.DataFrame(d)
            df_input["household_yes"] = 1 if household == "Yes" else 0
            df_input["sex_new_Male"] = 1 if sex == "Male" else 0

            for col in ['status_Active', 'status_Closed', 'status_Flagged', 'status_Outreach', 'status_Pending']:
                df_input[col] = 1 if col == f"status_{status}" else 0

            df_input["latest_language_is_english_Yes"] = 1 if latest_lang_english == "Yes" else 0

            for col in ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter']:
                df_input[col] = 1 if col == f"Season_{season}" else 0

            for col in [
                'Month_April', 'Month_August', 'Month_December', 'Month_Febuary', 'Month_January',
                'Month_July', 'Month_June', 'Month_March', 'Month_May', 'Month_November',
                'Month_October', 'Month_September']:
                df_input[col] = 1 if col == f"Month_{month}" else 0

            prediction = model.predict(df_input)[0]
            probs = model.predict_proba(df_input)[0]
            prob_return = probs[1]
            prob_not_return = probs[0]

            st.markdown("---")
            st.subheader("Prediction:")
            if prediction == 1:
                st.success("✅ Client is likely to return")
            else:
                st.warning("⚠️ Client may not return")
            st.info(f"🔢 Probability of returning: **{prob_return:.2%}**")
            st.info(f"🔢 Probability of not returning: **{prob_not_return:.2%}**")
    else:
        st.info("Please select a season to continue.")

# ================== Feature Graph Tab ==================
elif page == "Feature Analysis Graphs":
    st.title("📊 Feature Analysis")
    st.image("Graphs/fiupdate.png", caption="Feature Importance", use_container_width=True)
    st.markdown("---")
    st.image("Graphs/waterfall.png", caption="Waterfall Plot", use_container_width=True)

# ================== Chatbot Page ==================
elif page == "Chatbot (RAG)":
    chatbot_page_rag()
