# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model (pipeline)
model = joblib.load("client_retention_model.pkl")

st.title("🔄 Client Retention Predictor")

# Layout: sidebar for tab selection
col1, col2 = st.columns([1, 4])

# Tab selection in first column
with col1:
    page = st.radio("Please select a tab", ("Client Retention Predictor", "Feature Analysis Graphs", "Chatbot"))

# Main content in second column
with col2:
    if page == "Client Retention Predictor":
        st.write("Predict whether a client is likely to return based on their profile.")

        # Step 1: Ask for Season first
        season = st.selectbox("Season", ["Select a season", "Spring", "Summer", "Fall", "Winter"])

        # Dictionary of season to months
        season_months = {
            'Spring': ['March', 'April', 'May'],
            'Summer': ['June', 'July', 'August'],
            'Fall': ['September', 'October', 'November'],
            'Winter': ['December', 'January', 'February']
        }

        # Only show months if a season is selected
        if season != "Select a season":
            month = st.selectbox("Month", season_months[season])

            # Input form
            with st.form("prediction_form"):
                contact_method = st.selectbox("Contact Method", ['phone', 'email', 'in-person'])
                household = st.selectbox("Household Type", ['single', 'family'])
                preferred_language = st.selectbox("Preferred Language", ['english', 'other'])
                sex = st.selectbox("Sex", ['male', 'female'])
                status = st.selectbox("Status", ['new', 'returning', 'inactive'])
                latest_lang_english = st.selectbox("Latest Language is English", ['yes', 'no'])

                age = st.slider("Age", 18, 100, 35)
                dependents_qty = st.number_input("Number of Dependents", 0, 10, 1)
                distance_km = st.number_input("Distance to Location (km)", 0.0, 50.0, 5.0)
                num_of_contact_methods = st.slider("Number of Contact Methods", 1, 5, 2)

                submitted = st.form_submit_button("Predict")

            if submitted:
                # DataFrame with raw inputs
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

                # Predict
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]

                st.markdown("---")
                st.subheader("Prediction Result:")
                if prediction == 1:
                    st.success(f"✅ Client is likely to return (Probability: {probability:.2%})")
                else:
                    st.warning(f"⚠️ Client may not return (Probability: {probability:.2%})")
        else:
            st.info("Please select a season to continue.")

    elif page == "Feature Analysis Graphs":
        st.write("Feature Importance Plot")
        image_path = "Graphs/fiupdate.png"
        st.image(image_path, caption="Feature Importance", use_container_width=True)

        st.write("---")
        st.write("Waterfall Prediction Graph")
        image_path2 = "Graphs/waterfall.png"
        st.image(image_path2, caption="Waterfall Graph", use_container_width=True)

    elif page == "Chatbot":
        st.title("Chatbot")
        st.write("Chatbot feature coming soon.")
