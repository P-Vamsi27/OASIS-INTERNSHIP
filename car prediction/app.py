
import pandas as pd
import pickle as pk
import streamlit as st
model = pk.load(open(R"V:\OASIS INTERNSHIP\car prediction\model.pkl", "rb"))
st.header(" Car Price Prediction App")
cars_data = pd.read_csv(R"V:\OASIS INTERNSHIP\car prediction\vamsi_car_data.csv")
car_name = st.selectbox("Car Name", cars_data['Car_Name'].unique())
year = st.slider("Manufactured Year", 1994, 2024)
present_price = st.slider("Present Price (Lakhs)", 0.5, 50.0)
kms_driven = st.slider("Kilometers Driven", 100, 200000)
fuel = st.selectbox("Fuel Type", cars_data['Fuel_Type'].unique())
seller_type = st.selectbox("Seller Type", cars_data['Selling_type'].unique())
transmission = st.selectbox("Transmission", cars_data['Transmission'].unique())
owner = st.selectbox("Owner", cars_data['Owner'].unique())
if st.button("Predict"):
    input_df = pd.DataFrame([[present_price, kms_driven, owner, year, fuel, seller_type, transmission, car_name]],
                            columns=['Present_Price','Driven_kms','Owner','Year','Fuel_Type','Selling_type','Transmission','Car_Name'])
    input_df['Car_Age'] = 2026 - input_df['Year']
    input_df.drop('Year', axis=1, inplace=True)
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    predicted_price = model.predict(input_df)
    st.success(f" Predicted Selling Price: {round(predicted_price[0], 2)} Lakhs")
