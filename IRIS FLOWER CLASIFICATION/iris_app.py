# iris_app.py
import streamlit as st
import pandas as pd
import pickle as pk

# Load trained model
model = pk.load(open(r"V:\OASIS INTERNSHIP\IRIS FLOWER CLASIFICATION\iris_model.pkl", "rb"))

st.header(" Iris Flower Classification App")

# Sliders for input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 5.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0)
petal_width  = st.slider("Petal Width (cm)", 0.1, 3.0)

if st.button("Predict Species"):
    # Build input DataFrame with correct column names
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'])
    
    prediction = model.predict(input_df)[0]
    st.success(f" Predicted Species: {prediction}")
