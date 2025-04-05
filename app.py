import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# Load the Model and Chatbot Components
@st.cache_resource
def load_rag_components():
    df = pd.read_csv("chatbot_chunks.csv")  # This is your preprocessed dataset
    embeddings = np.load("chatbot_embeddings.npy")  # Precomputed embeddings for your data
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Pre-trained sentence transformer model
    return df, embeddings, embedder

# Retrieve context based on query (RAG)
def retrieve_context(query, embedder, doc_embeddings, top_k=5):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    scores = {}

    for idx, emb in enumerate(doc_embeddings):
        score = util.pytorch_cos_sim(query_emb, emb).item()
        scores[idx] = score

    # Sort and get the top k most relevant documents
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    context = "\n".join([docs['chunk'].iloc[idx] for idx, _ in top_docs])
    return context

# Query FLAN-T5 for Text Generation
def query_llm(query, context, llm):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    output = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].strip()

# Chatbot Page (RAG + FLAN-T5)
def chatbot_page():
    st.subheader("🧠 Chatbot (Data Q&A)")
    user_input = st.text_input("Ask your question:")

    if user_input:
        with st.spinner("Thinking..."):
            # Load RAG components (embeddings, dataset)
            df, embeddings, embedder = load_rag_components()

            # Load the FLAN-T5 model for question generation
            llm = pipeline("text2text-generation", model="google/flan-t5-large")

            # Retrieve relevant context from the dataset based on user input
            context = retrieve_context(user_input, embedder, embeddings)  # Retrieve relevant context
            answer = query_llm(user_input, context, llm)  # Generate the answer using FLAN-T5

            # Display the context and the answer
            st.write("Retrieved Context:")
            st.write(context)

            st.write("Answer:")
            st.write(answer)

# Client Retention Prediction Page
def predictor_page():
    st.subheader("🧠 Client Retention Prediction")

    # Load the model (ensure XGB_model.jlib is in the same directory)
    model = joblib.load("XGB_model.jlib")

    # Define input fields for user to interact with (Ensure this matches your input features)
    household = st.selectbox("Does the client have a household?", ['Yes', 'No'])
    sex = st.selectbox("Sex of client", ['Male', 'Female'])
    dependents_qty = st.slider("Number of Dependents", min_value=0, max_value=10, value=3)
    distance_km = st.slider("Distance to Pickup (in km)", min_value=1, max_value=50, value=7)
    status = st.selectbox("Current Status", ['Active', 'Closed', 'Pending', 'Outreach', 'Flagged'])
    latest_language_english = st.selectbox("Is the language English?", ['Yes', 'No'])
    season = st.selectbox("Season of Pickup", ['Spring', 'Summer', 'Fall', 'Winter'])
    month = st.selectbox("Month of Pickup", ['January', 'February', 'March', 'April', 'May', 'June', 
                                            'July', 'August', 'September', 'October', 'November', 'December'])

    # Encoding input values into the appropriate format used by the model
    df_input = pd.DataFrame({
        'household_yes': [1 if household == 'Yes' else 0],
        'sex_newMale': [1 if sex == 'Male' else 0],
        'dependents_qty': [dependents_qty],
        'distance_km': [distance_km],
        # Encoding status as one-hot variables
        'status_Active': [1 if status == 'Active' else 0],
        'status_Closed': [1 if status == 'Closed' else 0],
        'status_Flagged': [1 if status == 'Flagged' else 0],
        'status_Outreach': [1 if status == 'Outreach' else 0],
        'status_Pending': [1 if status == 'Pending' else 0],
        # Encoding language as one-hot variables
        'latest_language_is_english_Yes': [1 if latest_language_english == 'Yes' else 0],
        # Encoding season as one-hot variables
        'Season_Spring': [1 if season == 'Spring' else 0],
        'Season_Summer': [1 if season == 'Summer' else 0],
        'Season_Fall': [1 if season == 'Fall' else 0],
        'Season_Winter': [1 if season == 'Winter' else 0],
        # Encoding month as one-hot variables
        'Month_January': [1 if month == 'January' else 0],
        'Month_February': [1 if month == 'February' else 0],
        'Month_March': [1 if month == 'March' else 0],
        'Month_April': [1 if month == 'April' else 0],
        'Month_May': [1 if month == 'May' else 0],
        'Month_June': [1 if month == 'June' else 0],
        'Month_July': [1 if month == 'July' else 0],
        'Month_August': [1 if month == 'August' else 0],
        'Month_September': [1 if month == 'September' else 0],
        'Month_October': [1 if month == 'October' else 0],
        'Month_November': [1 if month == 'November' else 0],
        'Month_December': [1 if month == 'December' else 0],
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
