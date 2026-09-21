import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Get user input for each feature
delivery_distance = st.slider('Delivery Distance (km)', 0.0, 50.0, 25.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 20, 10)
num_stops = st.slider('Number of Stops', 1, 10, 5)
vehicle_age = st.slider('Vehicle Age (years)', 0, 15, 5)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.slider('Package Weight (kg)', 0.1, 100.0, 10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', 5.0, 20.0, 12.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 150, 60)

# Create a DataFrame from user input
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

# Make prediction
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction[0] == 1:
        st.error(f"Prediction: Delivery **Likely Delayed** (Probability: {prediction_proba[1]:.2f})")
    else:
        st.success(f"Prediction: Delivery **On Time** (Probability: {prediction_proba[0]:.2f})")

    st.write("Feature values:")
    st.write(input_data)

