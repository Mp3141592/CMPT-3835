import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ------------------------------------------------------------------------------
# Load Machine Learning Model
# ------------------------------------------------------------------------------
model = joblib.load("client_retention_model.pkl")

# ------------------------------------------------------------------------------
# Load Chatbot CSV Content
# ------------------------------------------------------------------------------
df_chunks = pd.read_csv("chatbot_chunks_combined (1).csv", header=None)
df_chunks.columns = ['chunk']  # assign column name if missing

documents = dict(enumerate(df_chunks["chunk"]))

# ------------------------------------------------------------------------------
# Setup Embedding Model
# ------------------------------------------------------------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

doc_embeddings = {
    doc_id: embedder.encode(text, convert_to_tensor=True)
    for doc_id, text in documents.items()
}

def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {
        doc_id: util.pytorch_cos_sim(query_embedding, emb).item()
        for doc_id, emb in doc_embeddings.items()
    }
    top_doc_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return "\n\n".join([documents[doc_id] for doc_id in top_doc_ids])

# ------------------------------------------------------------------------------
# Setup FLAN-T5 Generator
# ------------------------------------------------------------------------------
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-large")

generator = load_generator()

def query_llm(query, context):
    prompt = (
        "You are a helpful assistant that analyzes client retention insights.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query}\n\n"
        "Answer:"
    )
    result = generator(prompt, max_new_tokens=150, temperature=0.7)[0]['generated_text']
    return result.replace(prompt, "").strip()

# ------------------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Client Retention Dashboard", layout="wide")
st.title("🔄 Client Retention Predictor & 🤖 Chatbot Assistant")

col1, col2 = st.columns([1, 4])
with col1:
    page = st.radio("Choose a Section", ["Client Retention Predictor", "Feature Analysis Graphs", "Chatbot"])

with col2:
    # Page 1: Predictor
    if page == "Client Retention Predictor":
        st.subheader("📊 Predict if a Client Will Return")

        with st.form("prediction_form"):
            contact_method = st.selectbox("Contact Method", ['phone', 'email', 'in-person'])
            household = st.selectbox("Household Type", ['single', 'family'])
            preferred_language = st.selectbox("Preferred Language", ['english', 'other'])
            sex = st.selectbox("Sex", ['male', 'female'])
            status = st.selectbox("Status", ['new', 'returning', 'inactive'])
            season = st.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])
            month = st.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June',
                                           'July', 'August', 'September', 'October', 'November', 'December'])
            latest_lang_english = st.selectbox("Latest Language is English", ['yes', 'no'])
            age = st.slider("Age", 18, 100, 35)
            dependents_qty = st.number_input("Number of Dependents", 0, 10, 1)
            distance_km = st.number_input("Distance to Location (km)", 0.0, 50.0, 5.0)
            num_of_contact_methods = st.slider("Number of Contact Methods", 1, 5, 2)

            submitted = st.form_submit_button("Predict")

        if submitted:
            input_df = pd.DataFrame([{
                'contact_method': contact_method,
                'household': household,
                'preferred_languages': preferred_language,
                'sex_new': sex,
                'status': status,
                'Season': season,
                'Month': month,
                'latest_language_is_english': latest_lang_english,
                'age': age,
                'dependents_qty': dependents_qty,
                'distance_km': distance_km,
                'num_of_contact_methods': num_of_contact_methods
            }])

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("---")
            st.subheader("🔎 Prediction Result:")
            if prediction == 1:
                st.success(f"✅ Client is likely to return (Probability: {round(probability, 2)})")
            else:
                st.warning(f"⚠️ Client may not return (Probability: {round(probability, 2)})")

    # Page 2: Feature Importance Graphs
    elif page == "Feature Analysis Graphs":
        st.subheader("📈 Feature Importance")
        st.image("Graphs/fiupdate.png", caption="Top Features by Importance", use_container_width=True)
        st.markdown("---")
        st.subheader("📉 Waterfall Prediction Example")
        st.image("Graphs/waterfall.png", caption="How features contributed to the prediction", use_container_width=True)

    # Page 3: Chatbot
    elif page == "Chatbot":
        st.subheader("🤖 Ask the Client Insights Chatbot")
        user_query = st.text_input("Ask about pickups, languages, age, or client status:")

        if user_query:
            with st.spinner("Analyzing..."):
                context = retrieve_context(user_query)
                response = query_llm(user_query, context)
                st.success(response)
