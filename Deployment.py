# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:46:35 2025

@author: leekai

Note: UNTESTED
"""
import joblib as job

# Load models
rf_model = job.load("random_forest_model.pkl")
svr_model = job.load("svr_model.pkl")
xgb_model = job.load("xgb_model.pkl")

# Load scalers
scaler_X = job.load("scaler_X.pkl")
scaler_y = job.load("scaler_y.pkl")

# --- GETTING REAL TIME DATA ---

import time

# placeholder
from x_sensor_library import read_pH, read_conductivity, read_turbidity
from another_library import read_temperature, read_pressure, read_tds

# Function to read sensor data

def get_sensor_readings():
    try:
        pH = read_pH()                           # e.g., reads pH from sensor
        conductivity = read_conductivity()       # e.g., reads conductivity in uS/cm
        temperature = read_temperature()         # e.g., reads temperature in °C
        turbidity = read_turbidity()             # e.g., reads turbidity in NTU
        atmospheric_pressure = read_pressure()   # e.g., reads atmospheric pressure in hPa
        tds = read_tds()                         # e.g., reads Total Dissolved Solids (mg/L)

        # Combine all sensor readings into a list
        return [pH, conductivity, temperature, turbidity, atmospheric_pressure, tds]
    except Exception as e:
        print(f"Error reading sensors: {e}")
        return None

# Prediction loop
while True:
    try:
        # Get real-time sensor readings
        real_time_data = get_sensor_readings()
        
        if real_time_data:
            print(f"Real-Time Sensor Data: {real_time_data}")

            # Prediction process
            real_time_data_scaled = scaler_X.transform([real_time_data])  # Scale input data

            # Random Forest Prediction
            rf_pred_scaled = rf_model.predict(real_time_data_scaled)
            rf_pred = scaler_y.inverse_transform([rf_pred_scaled])[0]

            # SVR Prediction
            svr_pred_scaled = svr_model.predict(real_time_data_scaled)
            svr_pred = scaler_y.inverse_transform([svr_pred_scaled])[0]

            # XGBoost Prediction
            xgb_pred_scaled = xgb_model.predict(real_time_data_scaled)
            xgb_pred = scaler_y.inverse_transform([xgb_pred_scaled])[0]

            # Display predictions
            print(f"Random Forest Predicted DO: {rf_pred:.2f} mg/L")
            print(f"SVR Predicted DO: {svr_pred:.2f} mg/L")
            print(f"XGBoost Predicted DO: {xgb_pred:.2f} mg/L")

        # Delay before next reading (adjust based on your requirements)
        time.sleep(2) # Counts in seconds

    except KeyboardInterrupt:
        print("Exiting...")
        break
    except Exception as e:
        print(f"Error in prediction loop: {e}")
