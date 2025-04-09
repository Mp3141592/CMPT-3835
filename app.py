# app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the new pipeline model
model = joblib.load("client_retention_model.pkl")

st.title("🔄 Client Retention Predictor")

col1, col2 = st.columns([1, 4])  # Adjust the width ratio as needed

# Place the radio buttons in the first column
with col1:
    page = st.radio("Please select a tab", ("Client Retention Predictor", "Feature Analysis Graphs", "Chatbot"))

# Use the second column to display the content based on the selection
with col2:

    if page == "Client Retention Predictor":

        st.write("Predict whether a client is likely to return based on their profile.")

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
                # Pass raw data (no one-hot encoding)
                new_data = pd.DataFrame([{
                    'age': age,
                    'dependents_qty': dependents_qty,
                    'distance_km': distance_km,
                    'num_of_contact_methods': num_of_contact_methods,
                    'household': household,
                    'sex': sex,
                    'status': status,
                    'latest_language_is_english': latest_lang_english,
                    'Season': season,
                    'Month': month
                }])

                # Predict
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
        image_path = "Graphs/fiupdate.png"
        st.image(image_path, caption="Feature Importance", use_container_width=True)

        st.write("---")
        st.write("Waterfall Prediction Graph")
        image_path2 = "Graphs/waterfall.png"
        st.image(image_path2, caption="Waterfall Graph", use_container_width=True)

    elif page == "Chatbot":
        st.title("Chatbot")
