import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ===============================
# Load Chatbot Components
# ===============================
@st.cache_resource
def load_rag_components():
    df = pd.read_csv("chatbot_chunks.csv")
    embeddings = np.load("chatbot_embeddings.npy")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = {i: torch.tensor(embeddings[i]) for i in range(len(df))}
    llm = pipeline("text2text-generation", model="google/flan-t5-large")
    return df, embedder, doc_embeddings, llm

def retrieve_context(query, embedder, doc_embeddings, docs, top_k=5):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {
        idx: util.pytorch_cos_sim(query_emb, emb).item()
        for idx, emb in doc_embeddings.items()
    }
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return "\n".join(f"- {docs['chunk'][idx]}" for idx, _ in top_docs)

def query_llm(query, context, llm):
    prompt = (
        "Below are summaries of client data records.\n"
        "Use the information to answer the user's question clearly and accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    output = llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    return output[0]["generated_text"].replace(prompt, "").strip()

# ===============================
# Chatbot Page
# ===============================
def chatbot_page():
    st.title("🤖 Chatbot (Data Question Answering)")
    with st.spinner("Loading chatbot components..."):
        df, embedder, doc_embeddings, llm = load_rag_components()

    query = st.text_input("💬 Ask your question:")
    if st.button("Get Answer") and query:
        context = retrieve_context(query, embedder, doc_embeddings, df)
        answer = query_llm(query, context, llm)
        st.markdown("### 📄 Retrieved Context:")
        st.info(context)
        st.markdown("### 💬 Answer:")
        st.success(answer)

# ===============================
# Predictor Page
# ===============================
model = joblib.load("XGB_model.jlib")

def predictor_page():
    st.title("🔄 Client Retention Predictor")
    season = st.selectbox("Season of Pickup", ["Select", "Spring", "Summer", "Fall", "Winter"])
    season_months = {
        'Spring': ['March', 'April', 'May'],
        'Summer': ['June', 'July', 'August'],
        'Fall': ['September', 'October', 'November'],
        'Winter': ['December', 'January', 'February']
    }

    if season != "Select":
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
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success("✅ Client is likely to return")
            else:
                st.warning("⚠️ Client may not return")
            st.info(f"Probability of returning: {probs[1]*100:.2f}%")
            st.info(f"Probability of not returning: {probs[0]*100:.2f}%")
    else:
        st.info("Please select a season to continue.")

# ===============================
# Graphs Page
# ===============================
def graphs_page():
    st.title("📊 Feature Analysis Graphs")
    st.image("Graphs/fiupdate.png", caption="Feature Importance", use_container_width=True)
    st.image("Graphs/waterfall.png", caption="Waterfall Plot", use_container_width=True)

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("📂 Client Project")
page = st.sidebar.radio("Select a Page", [
    "Client Retention Predictor",
    "Feature Analysis Graphs",
    "Chatbot (Data Q&A)"
])

if page == "Client Retention Predictor":
    predictor_page()
elif page == "Feature Analysis Graphs":
    graphs_page()
elif page == "Chatbot (Data Q&A)":
    chatbot_page()
