import streamlit as st
import pandas as pd
import joblib

# Load model & preprocessing files
model = joblib.load("LR_car.pkl")
scaler = joblib.load("scaler.pkl")
ex_columns = joblib.load("columns.pkl")

st.title("Ford Car Price Prediction App")
st.write("Enter the car details below to estimate the selling price.")

# Input Fields
model_input = st.selectbox("Model", [
    'Fiesta','Focus','Kuga','EcoSport','C-MAX','Ka+','Mondeo','B-MAX','S-MAX',
    'Grand C-MAX','Galaxy','Edge','KA','Puma','Tourneo Custom','Mustang',
    'Grand Tourneo Connect','Tourneo Connect','Fusion','Streetka','Ranger',
    'Escort','Transit Tourneo'
])

year = st.slider("Year", 1999, 2026, 2017)

transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])

mileage = st.number_input("Mileage", 1, 200000, 30000)

fuelType = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])

tax = st.number_input("Tax", 0, 600, 150)

mpg = st.number_input("MPG", 10.0, 250.0, 50.0, step=0.1)

engineSize = st.selectbox("Engine Size", [
    0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0,
    2.2, 2.3, 2.5, 3.2, 5.0
])

if st.button("Predict Price"):
    # Create DataFrame
    data = {
        "model": model_input,
        "year": year,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuelType,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engineSize
    }

    df = pd.DataFrame([data])

    # One-hot encode
    df = pd.get_dummies(df)

    # Add missing columns
    for col in ex_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match model
    df = df[ex_columns]

    # Scale + Predict
    scaled = scaler.transform(df)
    predicted_price = model.predict(scaled)[0]

    st.success(f"Predicted Car Price: ₹{predicted_price:,.2f}")
