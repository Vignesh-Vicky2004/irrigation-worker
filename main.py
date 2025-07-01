import os
import json
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

# === Firebase Initialization ===
firebase_key_json = os.environ["FIREBASE_KEY_JSON"]
firebase_cred_dict = json.loads(firebase_key_json)
cred = credentials.Certificate(firebase_cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agri-hub-544be-default-rtdb.firebaseio.com'
})

# === Model Loading ===
MODEL_PATH = "tamil_nadu_irrigation_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']

# === FastAPI App ===
app = FastAPI()

# === Data Model ===
class SensorData(BaseModel):
    humidity: float
    temperature: float
    soilMoisture: float

# === Prediction Logic ===
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

        district_enc = encoders['le_district'].transform([full_input['district']])[0]
        zone_enc = encoders['le_zone'].transform([full_input['zone']])[0]
        season_enc = encoders['le_season'].transform([full_input['season']])[0]

        heat_stress = int(full_input['temperature_celsius'] > 35 and full_input['humidity_percent'] < 50)
        drought_stress = int(full_input['soil_moisture_percent'] < 30 and full_input['rainfall_mm_prediction_next_1h'] < 1)
        soil_temp_interaction = full_input['soil_moisture_percent'] * full_input['temperature_celsius']
        humidity_rain_interaction = full_input['humidity_percent'] * full_input['rainfall_mm_prediction_next_1h']

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

        timestamp = datetime.now().isoformat()
        db.reference('sensorData/prediction_class').set(irrigation_class)
        db.reference('sensorData/last_prediction_time').set(timestamp)

        print(f"✅ Prediction updated: Class {irrigation_class} at {timestamp}")

        return {"irrigation_class": irrigation_class, "timestamp": timestamp}
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return {"error": str(e)}

# === API Endpoints ===
@app.post("/predict")
def predict_route(data: SensorData):
    return predict_irrigation(data)

@app.get("/health")
def health_check():
    try:
        test_ref = db.reference("sensorData/raw")
        current_data = test_ref.get()
        return {
            "status": "healthy",
            "firebase_connected": True,
            "current_sensor_data": current_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "firebase_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/trigger-prediction")
def trigger_prediction():
    try:
        ref = db.reference("sensorData/raw")
        current_data = ref.get()
        if current_data:
            data = SensorData(
                humidity=float(current_data.get("humidity", 0.0)),
                temperature=float(current_data.get("temperature", 0.0)),
                soilMoisture=float(current_data.get("soilMoisture", 0.0))
            )
            result = predict_irrigation(data)
            return {"status": "success", "result": result, "input_data": current_data}
        else:
            return {"status": "error", "message": "No sensor data found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# === Real-time Firebase Listener ===
def setup_firebase_listener():
    def sensor_data_listener(event):
        try:
            print(f"🔔 Firebase listener triggered: {event.event_type}")
            print(f"📊 New data: {event.data}")
            if event.data and event.event_type in ['put', 'patch']:
                required_fields = ['humidity', 'temperature', 'soilMoisture']
                if all(field in event.data for field in required_fields):
                    data = SensorData(
                        humidity=float(event.data.get("humidity", 0.0)),
                        temperature=float(event.data.get("temperature", 0.0)),
                        soilMoisture=float(event.data.get("soilMoisture", 0.0))
                    )
                    result = predict_irrigation(data)
                    print(f"✅ Listener prediction result: {result}")
                else:
                    print(f"❌ Listener: Missing required fields in data: {event.data}")
        except Exception as e:
            print(f"❌ Firebase listener error: {e}")

    ref = db.reference("sensorData/raw")
    ref.listen(sensor_data_listener)
    print("🎧 Firebase real-time listener set up")

# === Startup Event ===
@app.on_event("startup")
def on_startup():
    print("🚀 Starting Firebase real-time listener for sensor data...")
    setup_firebase_listener()

