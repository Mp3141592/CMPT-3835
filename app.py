def predictor_page():
    model = joblib.load("client_retention_model.pkl")
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

        for col in ['status_Active', 'status_Closed', 'status_Flagged', 'status_Outreach', 'status_Pending']:
            df_input[col] = 1 if col == f"status_{status}" else 0

        df_input["latest_language_is_english_Yes"] = 1 if latest_lang_english == "Yes" else 0

        for col in ['Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter']:
            df_input[col] = 1 if col == f"Season_{season}" else 0

        for col in ['Month_April', 'Month_August', 'Month_December', 'Month_February', 'Month_January',
                    'Month_July', 'Month_June', 'Month_March', 'Month_May', 'Month_November',
                    'Month_October', 'Month_September']:
            df_input[col] = 1 if col == f"Month_{month}" else 0

        # Ensure all expected columns are present
        expected_columns = model.named_steps['preprocessor'].get_feature_names_out()
        missing = set(expected_columns) - set(df_input.columns)
        if missing:
            st.error(f"🚨 Missing columns in input: {missing}")
            return

        # Reindex to match model input
        df_input = df_input.reindex(columns=expected_columns, fill_value=0)

        prediction = model.predict(df_input)[0]
        probs = model.predict_proba(df_input)[0]

        st.subheader("🧠 Prediction Result:")
        if prediction == 1:
            st.success("✅ Client is likely to return")
        else:
            st.warning("⚠️ Client may not return")
        st.info(f"📈 Probability of returning: {probs[1]*100:.2f}%")
        st.info(f"📉 Probability of not returning: {probs[0]*100:.2f}%")
