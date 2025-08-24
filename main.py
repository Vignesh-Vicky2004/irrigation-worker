import os
import json
import threading
import time
import warnings
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

# Suppress sklearn warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ✅ Load Firebase credentials from env
firebase_key_json = os.environ["FIREBASE_KEY_JSON"]
firebase_cred_dict = json.loads(firebase_key_json)

cred = credentials.Certificate(firebase_cred_dict)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agri-hub-544be-default-rtdb.firebaseio.com'
})

# ✅ Load model
MODEL_PATH = "tamil_nadu_irrigation_model.pkl"
artifacts = joblib.load(MODEL_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']

# ✅ FastAPI app
app = FastAPI(title="Tamil Nadu Irrigation Prediction API", version="1.0.0")

# ✅ Data model
class SensorData(BaseModel):
    humidity: float
    temperature: float
    soilMoisture: float

# ✅ Root endpoint (fixes 404 error)
@app.get("/")
def root():
    return {
        "message": "Tamil Nadu Irrigation Prediction API", 
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict", 
            "trigger": "/trigger-prediction",
            "docs": "/docs"
        }
    }

# ✅ Prediction function (reused in both API and thread)
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

        # Update Firebase with timestamp
        timestamp = datetime.now().isoformat()
        db.reference('sensorData/prediction_class').set(irrigation_class)
        db.reference('sensorData/last_prediction_time').set(timestamp)
        
        print(f"✅ Prediction updated: Class {irrigation_class} at {timestamp}")

        return {"irrigation_class": irrigation_class, "timestamp": timestamp}
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return {"error": str(e)}

# ✅ API route
@app.post("/predict")
def predict_route(data: SensorData):
    return predict_irrigation(data)

# ✅ Improved background thread with better change detection
def monitor_firebase_sensor_data():
    last_sensor_values = None  # Only track sensor data, not metadata
    consecutive_errors = 0
    max_errors = 5

    print("🔄 Starting Firebase monitoring...")

    while True:
        try:
            ref = db.reference("sensorData")
            current = ref.get()
            
            if current is not None:
                # Extract only sensor data for comparison (ignore timestamps, predictions, etc.)
                current_sensor_data = {
                    'humidity': current.get('humidity'),
                    'temperature': current.get('temperature'), 
                    'soilMoisture': current.get('soilMoisture')
                }
                
                # Check if sensor data actually changed
                sensor_data_changed = (
                    last_sensor_values is None or 
                    current_sensor_data != last_sensor_values
                )
                
                if sensor_data_changed:
                    print("🔔 Detected change in SENSOR data!")
                    print(f"   Previous sensor data: {last_sensor_values}")
                    print(f"   Current sensor data:  {current_sensor_data}")
                    
                    # Validate data before processing
                    required_fields = ['humidity', 'temperature', 'soilMoisture']
                    if all(field in current_sensor_data and current_sensor_data[field] is not None for field in required_fields):
                        try:
                            data = SensorData(
                                humidity=float(current_sensor_data["humidity"]),
                                temperature=float(current_sensor_data["temperature"]),
                                soilMoisture=float(current_sensor_data["soilMoisture"])
                            )
                            result = predict_irrigation(data)
                            print(f"✅ Prediction result: {result}")
                            
                            # Update last_sensor_values after successful processing
                            last_sensor_values = current_sensor_data.copy()
                            consecutive_errors = 0  # Reset error counter
                            
                        except (ValueError, TypeError) as e:
                            print(f"❌ Data validation error: {e}")
                            print(f"   Raw sensor data: {current_sensor_data}")
                    else:
                        missing_fields = [f for f in required_fields if f not in current_sensor_data or current_sensor_data[f] is None]
                        print(f"❌ Missing/null required fields: {missing_fields}")
                        print(f"   Available sensor data: {current_sensor_data}")
                else:
                    print("📊 No change in sensor data (ignoring metadata updates)")
            else:
                print("⚠️  No sensor data found in Firebase")
                
        except Exception as e:
            consecutive_errors += 1
            print(f"❌ Error while monitoring sensor data (attempt {consecutive_errors}): {e}")
            
            if consecutive_errors >= max_errors:
                print(f"💥 Too many consecutive errors ({max_errors}). Stopping monitor.")
                break

        time.sleep(5)

# ✅ Start background monitoring
@app.on_event("startup")
def start_firebase_monitor():
    print("🚀 Starting Firebase monitoring...")
    threading.Thread(target=monitor_firebase_sensor_data, daemon=True).start()

# ✅ Health check endpoint (fixed path consistency)
@app.get("/health")
def health_check():
    try:
        # Test Firebase connection - use same path as monitoring
        test_ref = db.reference("sensorData")
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

# ✅ Manual trigger endpoint (fixed path consistency)
@app.post("/trigger-prediction")
def trigger_prediction():
    try:
        ref = db.reference("sensorData")  # Use consistent path
        current_data = ref.get()
        
        if current_data and all(field in current_data for field in ['humidity', 'temperature', 'soilMoisture']):
            data = SensorData(
                humidity=float(current_data.get("humidity", 0.0)),
                temperature=float(current_data.get("temperature", 0.0)),
                soilMoisture=float(current_data.get("soilMoisture", 0.0))
            )
            result = predict_irrigation(data)
            return {"status": "success", "result": result, "input_data": current_data}
        else:
            return {"status": "error", "message": "No valid sensor data found"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}
