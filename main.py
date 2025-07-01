# app/main.py

import os
import json
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Firebase Init from JSON string in ENV variable
firebase_key_json = os.environ["FIREBASE_KEY_JSON"]
firebase_cred_dict = json.loads(firebase_key_json)

cred = credentials.Certificate(firebase_cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agri-hub-544be-default-rtdb.firebaseio.com'
})

# ✅ Load Model
MODEL_PATH = "tamil_nadu_irrigation_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']

# ✅ FastAPI Init
app = FastAPI()

class SensorData(BaseModel):
    humidity: float
    temperature: float
    soilMoisture: float

@app.post("/predict")
def predict_irrigation(data: SensorData):
    try:
        now = datetime.now()
        full_input = {
            'soil_moisture_percent': data.soilMoisture,
            'temperature_celsius': data.temperature,
            'humidity_percent': data.humidity,
            'rainfall_mm_prediction_next_1h': 0.5,
            'hour': now.hour,
            'day_of_year': now.timetuple().tm_yday,
            'month': now.month,
            'district': 'Coimbatore',
            'zone': 'Western Zone',
            'season': 'southwest_monsoon'
        }

        # Encode categorical fields
        district_enc = encoders['le_district'].transform([full_input['district']])[0]
        zone_enc = encoders['le_zone'].transform([full_input['zone']])[0]
        season_enc = encoders['le_season'].transform([full_input['season']])[0]

        # Derived features
        heat_stress = int(full_input['temperature_celsius'] > 35 and full_input['humidity_percent'] < 50)
        drought_stress = int(full_input['soil_moisture_percent'] < 30 and full_input['rainfall_mm_prediction_next_1h'] < 1)
        soil_temp_interaction = full_input['soil_moisture_percent'] * full_input['temperature_celsius']
        humidity_rain_interaction = full_input['humidity_percent'] * full_input['rainfall_mm_prediction_next_1h']

        # Feature vector
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

        # Save result to Firebase
        db.reference('sensorData/prediction_class').set(irrigation_class)

        return {"irrigation_class": irrigation_class}

    except Exception as e:
        return {"error": str(e)}
