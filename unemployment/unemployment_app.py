# unemployment_app.py
import pandas as pd
import pickle as pk
import streamlit as st

# Load trained model
model = pk.load(open(r"V:\OASIS INTERNSHIP\unemployment\unemployment_model.pkl", "rb"))

st.header(" Unemployment Rate Prediction App")


# Load dataset for dropdown options
data = pd.read_csv(r"V:\OASIS INTERNSHIP\unemployment\Unemployment in India.csv")
data.columns = data.columns.str.strip()
region = st.selectbox("Select Region", data['Region'].unique())
year = st.slider("Year", 2019, 2026)
month = st.slider("Month", 1, 12)
frequency = st.selectbox("Frequency", data['Frequency'].unique())
area = st.selectbox("Area Type", data['Area'].unique())
estimated_employed = st.number_input("Estimated Employed", min_value=1000.0, max_value=50000000.0, step=1000.0)
labour_participation = st.slider("Labour Participation Rate (%)", 10.0, 80.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[region, frequency, estimated_employed, labour_participation, area, year, month]],
                            columns=['Region','Frequency','Estimated Employed','Estimated Labour Participation Rate (%)','Area','Year','Month'])
    
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    predicted_rate = model.predict(input_df)
    st.success(f" Predicted Unemployment Rate: {round(predicted_rate[0], 2)}%")
