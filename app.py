import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("delivery_model.pkl", "rb"))

st.title("🚴 Food Delivery Time Prediction")

st.write("Predict delivery time using Machine Learning")

distance = st.slider("Distance (km)", 1.0, 25.0)
prep_time = st.slider("Food Preparation Time (min)", 5, 40)
experience = st.slider("Courier Experience (years)", 0, 10)

weather = st.selectbox("Weather", ["Sunny","Rainy","Foggy","Windy"])
traffic = st.selectbox("Traffic Level", ["Low","Medium","High"])
time_day = st.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night"])
vehicle = st.selectbox("Vehicle Type", ["Bike","Scooter","Car"])

# Simple encoding
weather_dict = {"Sunny":0,"Rainy":1,"Foggy":2,"Windy":3}
traffic_dict = {"Low":0,"Medium":1,"High":2}
time_dict = {"Morning":0,"Afternoon":1,"Evening":2,"Night":3}
vehicle_dict = {"Bike":0,"Scooter":1,"Car":2}

if st.button("Predict Delivery Time"):

    input_data = np.array([[distance,
                            weather_dict[weather],
                            traffic_dict[traffic],
                            time_dict[time_day],
                            vehicle_dict[vehicle],
                            prep_time,
                            experience]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")