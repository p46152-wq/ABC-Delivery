import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Input features from the X.columns output
delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.0, max_value=100.0, value=25.0)
traffic_congestion = st.selectbox('Traffic Congestion (1=Low, 5=High)', [1, 2, 3, 4, 5], index=2)
weather_condition = st.selectbox('Weather Condition (1=Good, 5=Bad)', [1, 2, 3, 4, 5], index=1)
delivery_slot = st.selectbox('Delivery Slot (1=Morning, 2=Afternoon, 3=Evening)', [1, 2, 3], index=0)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, max_value=30, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, max_value=20, value=3)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, max_value=15, value=2)
road_condition_score = st.selectbox('Road Condition Score (1=Poor, 5=Excellent)', [1, 2, 3, 4, 5], index=2)
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, max_value=50.0, value=5.0)
fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, max_value=30.0, value=15.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, max_value=120, value=45)


if st.button('Predict Delivery Delay'):
    # Create a DataFrame from the inputs
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
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('**Delivery is LIKELY to be Delayed!**')
    else:
        st.success('**Delivery is LIKELY to be On Time!**')

    st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")
