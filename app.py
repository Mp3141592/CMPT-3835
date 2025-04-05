import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ===========================================
# Load Model for Prediction (XGBoost)
# ===========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("XGB_model.jlib")
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# ===========================================
# Load RAG Chatbot Components (Embeddings and Chunks)
# ===========================================
@st.cache_resource
def load_rag_components():
    df = pd.read_csv("chatbot_chunks_final.csv")  # Load the summarized data
    embeddings = np.load("chatbot_embeddings.npy")  # Load the precomputed embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model
    return df, embeddings, embedder

# ===========================================
# Retrieve Context Function (RAG)
# ===========================================
def retrieve_context(query, embedder, doc_embeddings, docs, top_k=5):
    query_emb = embedder.encode(query, convert_to_tensor=True)  # Get query embeddings
    scores = {}

    for idx, emb in enumerate(doc_embeddings):
        score = util.pytorch_cos_sim(query_emb, emb).item()  # Cosine similarity
        scores[idx] = score

    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]  # Get top_k results
    context = "\n".join([docs['chunk'].iloc[idx] for idx, _ in top_docs])  # Combine relevant chunks
    return context

# ===========================================
# Query FLAN-T5 for Text Generation
# ===========================================
def query_llm(query, context, llm):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    output = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].strip()  # Clean the output

# ===========================================
# Chatbot Function (RAG + FLAN-T5)
# ===========================================
def chatbot_page():
    st.subheader("🤖 Chatbot (Data Q&A)")
    user_input = st.text_input("Ask your question:")

    if user_input:
        with st.spinner("Thinking..."):
            df, embeddings, embedder = load_rag_components()  # Load the necessary components
            llm = pipeline("text2text-generation", model="google/flan-t5-large")  # FLAN-T5 for text generation

            context = retrieve_context(user_input, embedder, embeddings, df)  # Retrieve context based on the query
            answer = query_llm(user_input, context, llm)  # Generate an answer based on the context

            st.markdown("### 📄 Retrieved Context:")
            st.info(context)  # Show retrieved context

            st.markdown("### 💬 Answer:")
            st.success(answer)  # Show generated answer

# ===========================================
# Client Return Predictor
# ===========================================
def predictor_page():
    st.subheader("🔮 Client Return Predictor")

    # Step 1: Ask for Season first
    season = st.selectbox("Season of Pickup", ["Select a season", "Spring", "Summer", "Fall", "Winter"])

    # Dictionary of season to months
    season_months = {
        'Spring': ['March', 'April', 'May'],
        'Summer': ['June', 'July', 'August'],
        'Fall': ['September', 'October', 'November'],
        'Winter': ['December', 'January', 'February']
    }

    # Only show the month selection once a valid season is selected
    if season != "Select a season":
        month = st.selectbox("Month of Pickup", season_months[season])

        # Then show the rest of the form
        with st.form("prediction_form"):
            age = st.slider("Age", 18, 100, 35)
            dependents_qty = st.number_input("Number of Dependents", 0, 12, 1)
            distance_km = st.number_input("Distance to Location (km)", 0.0, 210.0, 5.0)
            num_of_contact_methods = st.slider("Number of Contact Methods", 1, 5, 2)
            household = st.selectbox("Has a household", ['Yes', 'No'])
            sex = st.selectbox("Gender of Client", ['Male', 'Female'])
            status = st.selectbox("Current Status", ['Active', 'Closed', 'Pending', 'Outreach', 'Flagged'])
            latest_lang_english = st.selectbox("Latest Language is English", ['Yes', 'No'])
            submitted = st.form_submit_button("Predict")

        if submitted:
            d = {
                'age': [age],
                'dependents_qty': [dependents_qty],
                'distance_km': [distance_km],
                'num_of_contact_methods': [num_of_contact_methods]
            }

            new_data = pd.DataFrame(d)

            # One-hot encodings
            new_data["household_yes"] = 1 if household == "Yes" else 0
            new_data["sex_new_Male"] = 1 if sex == "Male" else 0

            status_list = ['status_Active', 'status_Closed', 'status_Flagged', 'status_Outreach', 'status_Pending']
            for i in status_list:
                new_data[i] = 1 if i == f"status_{status}" else 0

            new_data["latest_language_is_english_Yes"] = 1 if latest_lang_english == "Yes" else 0

            season_list = ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter']
            for i in season_list:
                new_data[i] = 1 if i == f"Season_{season}" else 0

            month_list = [
                'Month_April', 'Month_August', 'Month_December', 'Month_Febuary', 'Month_January',
                'Month_July', 'Month_June', 'Month_March', 'Month_May', 'Month_November',
                'Month_October', 'Month_September'
            ]
            for i in month_list:
                new_data[i] = 1 if i == f"Month_{month}" else 0

            # Predict
            model = load_model()
            if model:
                prediction = model.predict(new_data)[0]
                probs = model.predict_proba(new_data)[0]
                prob_return = probs[1]
                prob_not_return = probs[0]

                st.markdown("---")
                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.success("✅ Client is likely to return")
                else:
                    st.warning("⚠️ Client may not return")

                st.info(f"🔢 Probability of returning: **{prob_return:.2%}**")
                st.info(f"🔢 Probability of not returning: **{prob_not_return:.2%}**")

    else:
        st.info("Please select a season to continue.")

# ===========================================
# Feature Analysis Graphs
# ===========================================
def graphs_page():
    st.subheader("📊 Feature Analysis Graphs")
    st.image("Graphs/fiupdate.png", caption="Feature Importance", use_container_width=True)
    st.image("Graphs/waterfall.png", caption="SHAP Waterfall Plot", use_container_width=True)

# ===========================================
# Main function
# ===========================================
def main():
    st.set_page_config(page_title="Client Retention Tool", layout="wide")
    st.title("📊 Client Retention Dashboard")

    # Navigation
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

# Run the app
if __name__ == "__main__":
    main()
