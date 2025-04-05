import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ==========================================
# Load Chatbot Components
# ==========================================
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
    return "\n- ".join([f"- {docs['chunk'].iloc[idx]}" for idx, _ in top_docs])

def query_llm(query, context, llm):
    prompt = (
        "Below are summaries of client data records.\n"
        "Use the information to answer the user’s question as clearly and insightfully as possible.\n"
        "If the question is general (e.g., 'what is this dataset about?'), summarize patterns.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    output = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].replace(prompt, "").strip()

# ==========================================
# Retention Prediction Page
# ==========================================
def predictor_page():
    model = joblib.load("XGB_model.jlib")
    st.subheader("🔄 Client Retention Predictor")

    with st.form("prediction_form"):
        age = st.slider("Age", 18, 100, 35)
        dependents_qty = st.number_input("Dependents", 0, 12, 1)
        distance_km = st.number_input("Distance to Location (km)", 0.0, 210.0, 5.0)
        num_of_contact_methods = st.slider("Number of Contact Methods", 1, 5, 2)
        household = st.selectbox("Has a household", ['Yes', 'No'])
        sex = st.selectbox("Gender", ['Male', 'Female'])
        status = st.selectbox("Client Status", ['Active', 'Closed', 'Pending', 'Outreach', 'Flagged'])
        latest_lang_english = st.selectbox("Language is English", ['Yes', 'No'])
        season = st.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])

        # Dynamically update months by season
        season_month_map = {
            'Spring': ['March', 'April', 'May'],
            'Summer': ['June', 'July', 'August'],
            'Fall': ['September', 'October', 'November'],
            'Winter': ['December', 'January', 'February']
        }
        month = st.selectbox("Month", season_month_map[season])
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

        for col in ['status_Active', 'status_Closed','status_Flagged', 'status_Outreach', 'status_Pending']:
            df_input[col] = 1 if col == f"status_{status}" else 0

        df_input["latest_language_is_english_Yes"] = 1 if latest_lang_english == "Yes" else 0

        for col in ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter']:
            df_input[col] = 1 if col == f"Season_{season}" else 0

        for col in ['Month_April', 'Month_August','Month_December', 'Month_Febuary', 'Month_January',
                    'Month_July', 'Month_June', 'Month_March', 'Month_May', 'Month_November',
                    'Month_October', 'Month_September']:
            df_input[col] = 1 if col == f"Month_{month}" else 0

        prediction = model.predict(df_input)[0]
        probs = model.predict_proba(df_input)[0]

        st.subheader("🧠 Prediction Result:")
        if prediction == 1:
            st.success("✅ Client is likely to return")
        else:
            st.warning("⚠️ Client may not return")
        st.info(f"📈 Probability of returning: {probs[1]*100:.2f}%")
        st.info(f"📉 Probability of not returning: {probs[0]*100:.2f}%")

# ==========================================
# Graphs Page
# ==========================================
def graphs_page():
    st.subheader("📊 Feature Analysis Graphs")
    st.image("Graphs/fiupdate.png", caption="Feature Importance", use_container_width=True)
    st.image("Graphs/waterfall.png", caption="SHAP Waterfall Plot", use_container_width=True)

# ==========================================
# Chatbot Page
# ==========================================
def chatbot_page():
    st.subheader("🤖 Chatbot (Data Q&A)")
    user_input = st.text_input("Ask your question:")

    if user_input:
        df, embedder, doc_embeddings, llm = load_rag_components()
        with st.spinner("Thinking..."):
            context = retrieve_context(user_input, embedder, doc_embeddings, df)
            st.markdown("#### 📄 Retrieved Context:")
            st.info(context)
            answer = query_llm(user_input, context, llm)
            st.markdown("#### 💬 Answer:")
            st.success(answer)

# ==========================================
# Streamlit Page Selector
# ==========================================
st.set_page_config(page_title="Client Retention App", layout="wide")
st.title("📈 Client Insights & Retention Dashboard")

page = st.sidebar.radio("Navigation", ["Client Retention Predictor", "Feature Analysis Graphs", "Chatbot (Data Q&A)"])

if page == "Client Retention Predictor":
    predictor_page()
elif page == "Feature Analysis Graphs":
    graphs_page()
elif page == "Chatbot (Data Q&A)":
    chatbot_page()
