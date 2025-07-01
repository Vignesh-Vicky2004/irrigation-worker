import os
import json
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import os

# 🔐 Step 1: Load Firebase key from ENV variable
FIREBASE_KEY_JSON = os.environ.get("FIREBASE_KEY_JSON")

if not FIREBASE_KEY_JSON:
    raise Exception("FIREBASE_KEY_JSON not found in environment variables")

with open("firebase_key.json", "w") as f:
    f.write(FIREBASE_KEY_JSON)

# 🔄 Step 2: Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://agri-hub-544be-default-rtdb.firebaseio.com'
    })

# 📦 Step 3: Load model
MODEL_PATH = "tamil_nadu_irrigation_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']
features = artifacts['feature_columns']

# 🔁 Step 4: Callback to handle predictions
def predict_and_update(event):
    try:
        input_data = event.data
        if input_data is None:
            return

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

        # Encode categoricals
        district_enc = encoders['le_district'].transform([full_input['district']])[0]
        zone_enc = encoders['le_zone'].transform([full_input['zone']])[0]
        season_enc = encoders['le_season'].transform([full_input['season']])[0]

        # Derived features
        heat_stress = int(full_input['temperature_celsius'] > 35 and full_input['humidity_percent'] < 50)
        drought_stress = int(full_input['soil_moisture_percent'] < 30 and full_input['rainfall_mm_prediction_next_1h'] < 1)
        soil_temp_interaction = full_input['soil_moisture_percent'] * full_input['temperature_celsius']
        humidity_rain_interaction = full_input['humidity_percent'] * full_input['rainfall_mm_prediction_next_1h']

        # Create final vector
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

        scaled_input = scaler.transform(feature_vector)
        irrigation_class = int(model.predict(scaled_input)[0])

        # Push result back to Firebase
        db.reference('sensorData/prediction_class').set(irrigation_class)
        print(f"✅ Sent irrigation_class = {irrigation_class}")

    except Exception as e:
        print(f"❌ Error: {e}")

# 🚀 Step 5: Listen forever
if __name__ == "__main__":
    print("🚀 Firebase Irrigation Worker started on Render")
    sensor_ref = db.reference("sensorData")
    sensor_ref.listen(predict_and_update)
