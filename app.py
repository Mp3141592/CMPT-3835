# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import faiss
from sentence_transformers import SentenceTransformer

# === Load trained XGBoost model ===
model = joblib.load("XGB_model.jlib")

# === Load chatbot data and FAISS ===
@st.cache_resource
def load_chatbot():
    docs = pd.read_csv("chatbot_chunks.csv")  # must contain a 'chunk' column
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embedder.encode(docs['chunk'].tolist(), convert_to_numpy=True)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    return docs, embedder, index

# === Main UI ===
st.title("🔄 Client Retention Predictor")

col1, col2 = st.columns([1, 4])

with col1:
    page = st.radio("Please select a tab", ("Client Retention Predictor", "Feature Analysis Graphs", "Chatbot"))

with col2:
    if page == "Client Retention Predictor":
        st.write("Predict whether a client is likely to return based on their profile.")

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

    elif page == "Feature Analysis Graphs":
        st.write('Feature Importance Plot')
        st.image("Graphs/fiupdate.png", caption="Feature Importance", use_container_width=True)
        st.write("---")
        st.write("Waterfall Prediction Graph")
        st.image("Graphs/waterfall.png", caption="Waterfall Graph", use_container_width=True)

    elif page == "Chatbot":
        st.title("🤖 Chatbot: Ask About Client Data")

        docs, embedder, index = load_chatbot()

        user_query = st.text_input("💬 Ask your question here:")
        if st.button("Get Answer") and user_query:
            try:
                query_embedding = embedder.encode([user_query])
                D, I = index.search(np.array(query_embedding), k=3)
                results = docs.iloc[I[0]]['chunk'].tolist()

                st.markdown("### 🧠 Top Matching Responses:")
                for i, chunk in enumerate(results, 1):
                    st.markdown(f"**{i}.** {chunk}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
