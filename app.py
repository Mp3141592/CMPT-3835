# app.py
import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from datetime import datetime

# Load the trained model
model = joblib.load("XGB_model.jlib")

st.title("🔄 Client Retention Predictor")

col1, col2 = st.columns([1, 4])  # Adjust the width ratio as needed

# Place the radio buttons in the first column
with col1:
    page = st.radio("Please select a tab", ("Client Retention Predictor", "Feature Analysis Graphs", "Chatbot"))

# Use the second column to display the content based on the selection
with col2:

    if page == "Client Retention Predictor":
        
        st.write("Predict whether a client is likely to return based on their profile.")

        # Input form
        with st.form("prediction_form"):
            age = st.slider("Age", 18, 100, 35)
            dependents_qty = st.number_input("Number of Dependents", 0, 12, 1)
            distance_km = st.number_input("Distance to Location (km)", 0.0, 210.0, 5.0)
            num_of_contact_methods = st.slider("Number of Contact Methods", 1, 5, 2)
            household = st.selectbox("Has a household", ['Yes', 'No'])
            sex = st.selectbox("Gender of Client", ['Male', 'Female'])
            status = st.selectbox("Current Status", ['Active', 'Closed', 'Pending', 'Outreach', 'Flagged'])
            latest_lang_english = st.selectbox("Latest Language is English", ['Yes', 'No'])

            # Season and dynamic month selection
            season = st.selectbox("Season of Pickup", ['Spring', 'Summer', 'Fall', 'Winter'])

            season_months = {
                'Spring': ['March', 'April', 'May'],
                'Summer': ['June', 'July', 'August'],
                'Fall': ['September', 'October', 'November'],
                'Winter': ['December', 'January', 'February']
            }

            month = st.selectbox("Month of Pickup", season_months[season])

            submitted = st.form_submit_button("Predict")

        # Prepare input and predict
        if submitted:
            
            d = {
                'age': [age],
                'dependents_qty': [dependents_qty],
                'distance_km': [distance_km],
                'num_of_contact_methods': [num_of_contact_methods]
            }

            new_data = pd.DataFrame(d)

            # Household
            new_data["household_yes"] = 1 if household == "Yes" else 0

            # Sex
            new_data["sex_new_Male"] = 1 if sex == "Male" else 0

            # Status one-hot encoding
            status_list = ['status_Active', 'status_Closed', 'status_Flagged', 'status_Outreach', 'status_Pending']
            for i in status_list:
                new_data[i] = 1 if i == f"status_{status}" else 0

            # Latest language
            new_data["latest_language_is_english_Yes"] = 1 if latest_lang_english == "Yes" else 0

            # Season one-hot encoding
            season_list = ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter']
            for i in season_list:
                new_data[i] = 1 if i == f"Season_{season}" else 0

            # Month one-hot encoding
            month_list = [
                'Month_April', 'Month_August', 'Month_December', 'Month_Febuary', 'Month_January',
                'Month_July', 'Month_June', 'Month_March', 'Month_May', 'Month_November',
                'Month_October', 'Month_September'
            ]
            for i in month_list:
                new_data[i] = 1 if i == f"Month_{month}" else 0

            # Make prediction
            prediction = model.predict(new_data)[0]

            st.markdown("---")
            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"✅ Client is likely to return")
            else:
                st.warning(f"⚠️ Client may not return")

    elif page == "Feature Analysis Graphs":
        st.write('Feature Importance Plot')

        image_path = "Graphs/fiupdate.png"
        st.image(image_path, caption="Feature Importance", use_container_width=True)

        st.write("---")
        st.write("Waterfall Prediction Graph")

        image_path2 = "Graphs/waterfall.png"
        st.image(image_path2, caption="Waterfall Graph", use_container_width=True)

    elif page == "Chatbot":
        st.title("Chatbot")
