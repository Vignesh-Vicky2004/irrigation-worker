import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os

# 🔐 Step 1: Load Firebase credentials (Render injects this via Secret File)
FIREBASE_KEY_PATH = "/opt/render/project/src/firebase_key.json"

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://agri-hub-544be-default-rtdb.firebaseio.com'
    })

# 📦 Step 2: Load ML model and encoders
MODEL_PATH = "tamil_nadu_irrigation_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']
features = artifacts['feature_columns']

# 🔁 Step 3: Prediction function
def predict_and_update(input_data):
    try:
        print(f"\n📥 Received sensorData: {input_data}")

        humidity = input_data.get('humidity')
        temperature = input_data.get('temperature')
        soil_moisture = input_data.get('soilMoisture')

        if None in (humidity, temperature, soil_moisture):
            print("❌ Missing required fields")
            return

        now = datetime.now()
        full_input = {
            'soil_moisture_percent': float(soil_moisture),
            'temperature_celsius': float(temperature),
            'humidity_percent': float(humidity),
            'rainfall_mm_prediction_next_1h': 0.5,
            'hour': now.hour,
            'day_of_year': now.timetuple().tm_yday,
            'month': now.month,
            'district': 'Coimbatore',
            'zone': 'Western Zone',
            'season': 'southwest_monsoon'
        }

        # Encode categorical values
        district_enc = encoders['le_district'].transform([full_input['district']])[0]
        zone_enc = encoders['le_zone'].transform([full_input['zone']])[0]
        season_enc = encoders['le_season'].transform([full_input['season']])[0]

        # Derived features
        heat_stress = int(full_input['temperature_celsius'] > 35 and full_input['humidity_percent'] < 50)
        drought_stress = int(full_input['soil_moisture_percent'] < 30 and full_input['rainfall_mm_prediction_next_1h'] < 1)
        soil_temp_interaction = full_input['soil_moisture_percent'] * full_input['temperature_celsius']
        humidity_rain_interaction = full_input['humidity_percent'] * full_input['rainfall_mm_prediction_next_1h']

        # Create input vector
        feature_vector = np.array([[
            full_input['soil_moisture_percent'],
            full_input['temperature_celsius'],
            full_input['humidity_percent'],
            full_input['rainfall_mm_prediction_next_1h'],
            full_input['hour'],
            full_input['day_of_year'],
            full_input['month'],
            district_enc,
            zone_enc,
            season_enc,
            heat_stress,
            drought_stress,
            soil_temp_interaction,
            humidity_rain_interaction
        ]])

        df_input = pd.DataFrame(feature_vector, columns=features)
        scaled_input = scaler.transform(df_input)
        irrigation_class = int(model.predict(scaled_input)[0])

        # Push prediction to Firebase
        db.reference('sensorData/prediction_class').set(irrigation_class)
        print(f"✅ Sent irrigation_class = {irrigation_class}")

    except Exception as e:
        print(f"❌ Error: {e}")

# 🔄 Step 4: Polling loop to watch sensorData for changes
def poll_sensor_data(interval=5):
    print("🚀 Polling sensorData for updates every", interval, "seconds")
    last_snapshot = None
    while True:
        try:
            snapshot = db.reference("sensorData").get()
            if snapshot and snapshot != last_snapshot:
                last_snapshot = snapshot
                predict_and_update(snapshot)
        except Exception as e:
            print(f"❌ Polling error: {e}")
        time.sleep(interval)

# 🚀 Start the polling worker
if __name__ == "__main__":
    poll_sensor_data()
