# app.py

import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ------------------------------------------------------------------------------
# Load Model for Prediction Tab
# ------------------------------------------------------------------------------
model = joblib.load("client_retention_model.pkl")

# ------------------------------------------------------------------------------
# Load Data for Chatbot
# ------------------------------------------------------------------------------
df_chunks = pd.read_csv("chatbot_chunks_combined_improved (version 1).csv")
documents = df_chunks["chunk"]

# Embed documents
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = {
    doc_id: embedder.encode(text, convert_to_tensor=True)
    for doc_id, text in documents.items()
}

# FLAN-T5 chatbot pipeline
generator = pipeline("text2text-generation", model="google/flan-t5-large")

def retrieve_context(query, top_k=2):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = {
        doc_id: util.pytorch_cos_sim(query_embedding, emb).item()
        for doc_id, emb in doc_embeddings.items()
    }
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, _ in sorted_docs[:top_k]]
    return "\n\n".join(documents[doc_id] for doc_id in top_doc_ids)

def query_llm(query, context):
    prompt = (
        "You have some background info below. Answer the user's query clearly.\n\n"
        f"Context:\n{context}\n\nUser Query: {query}\n\nAnswer:"
    )
    response = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']
    return response.strip().replace(prompt, '').strip()

def rag_chatbot(query):
    context = retrieve_context(query)
    return query_llm(query, context)

# ------------------------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------------------------
st.title("🔄 Client Retention Predictor + Chatbot")

col1, col2 = st.columns([1, 4])
with col1:
    page = st.radio("Choose a tab", ("Client Retention Predictor", "Feature Analysis Graphs", "Chatbot"))

with col2:

    # ---------------------- TAB 1: Prediction ----------------------
    if page == "Client Retention Predictor":
        st.write("Predict whether a client is likely to return based on their profile.")

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
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"✅ Client is likely to return (Probability: {round(probability, 2)})")
            else:
                st.warning(f"⚠️ Client may not return (Probability: {round(probability, 2)})")

    # ---------------------- TAB 2: Feature Analysis ----------------------
    elif page == "Feature Analysis Graphs":
        st.write('Feature Importance Plot')
        st.image("Graphs/fiupdate.png", caption="Feature Importance", use_container_width=True)
        st.write("---")
        st.write("Waterfall Prediction Graph")
        st.image("Graphs/waterfall.png", caption="Waterfall Graph", use_container_width=True)

    # ---------------------- TAB 3: Chatbot ----------------------
    elif page == "Chatbot":
        st.write("Ask a question based on client data or operations. Example:")
        st.info("🧠 Example: *What trends can be observed from client return patterns?*")
        user_query = st.text_input("Type your question here")

        if user_query:
            with st.spinner("Thinking..."):
                answer = rag_chatbot(user_query)
                st.markdown("**Answer:**")
                st.write(answer)
