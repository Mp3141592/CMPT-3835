# app.py

import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load the trained pipeline model
model = joblib.load("client_retention_model.pkl")

# Initialize sentence transformer and text generator
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text2text-generation", model="google/flan-t5-large")

# Sample transaction data
transaction_data = pd.DataFrame({
    "Client_ID": [101, 102, 103],
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [29, 34, 42],
    "Pickup_Date": ["2023-01-15", "2023-02-15", "2023-03-15"],
    "Hamper_Type": ["Standard", "Premium", "Standard"],
    "Location": ["Downtown", "Uptown", "Midtown"]
})

# Generate narrative string
transaction_narrative = "Here are the latest client transactions:\n"
for idx, row in transaction_data.iterrows():
    transaction_narrative += (
        f"Client {row['Client_ID']} ({row['Name']}, Age {row['Age']}) picked "
        f"up a {row['Hamper_Type']} hamper at {row['Location']} on {row['Pickup_Date']}.\n"
    )

# Static documents
documents = {
    "doc1": (
        "XYZ Charity is a non-profit organization focused on distributing food hampers. "
        "It aims to improve community well-being by providing support to families in need."
    ),
    "doc2": transaction_narrative
}

# Precompute document embeddings
doc_embeddings = {
    doc_id: embedder.encode(text, convert_to_tensor=True)
    for doc_id, text in documents.items()
}

# Streamlit app layout
st.title("🔄 Client Retention Predictor")

col1, col2 = st.columns([1, 4])
with col1:
    page = st.radio("Please select a tab", ("Client Retention Predictor", "Feature Analysis Graphs", "Chatbot"))

with col2:

    # -----------------------------------------
    # Tab 1: Client Retention Prediction
    # -----------------------------------------
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

    # -----------------------------------------
    # Tab 2: Feature Analysis Graphs
    # -----------------------------------------
    elif page == "Feature Analysis Graphs":
        st.write('📊 Feature Importance Plot')
        st.image("Graphs/fiupdate.png", caption="Feature Importance", use_container_width=True)

        st.write("---")
        st.write("📉 Waterfall Prediction Graph")
        st.image("Graphs/waterfall.png", caption="Waterfall Graph", use_container_width=True)

    # -----------------------------------------
    # Tab 3: Chatbot
    # -----------------------------------------
    elif page == "Chatbot":
        st.subheader("💬 Ask the Chatbot")
        st.write("Ask anything about the charity or recent client transactions.")

        query = st.text_input("Type your question:")

        def retrieve_context(query, top_k=2):
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            scores = {
                doc_id: util.pytorch_cos_sim(query_embedding, emb).item()
                for doc_id, emb in doc_embeddings.items()
            }
            top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return "\n\n".join(documents[doc_id] for doc_id, _ in top_docs)

        def query_llm(query, context):
            prompt = (
                "You have some background info plus transaction data below. "
                "Analyze the context and answer the user’s query clearly and succinctly.\n\n"
                f"Context:\n{context}\n\n"
                f"User Query: {query}\n\n"
                "Answer:"
            )
            result = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
            raw_output = result[0]['generated_text']
            return raw_output.replace(prompt, "").strip() if raw_output.startswith(prompt) else raw_output.strip()

        def rag_chatbot(query):
            context = retrieve_context(query)
            return query_llm(query, context)

        if query:
            response = rag_chatbot(query)
            st.markdown("#### 🤖 Chatbot's Response")
            st.write(response)
