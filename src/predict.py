import joblib
import numpy as np

# Load models
model = joblib.load("../models/model.pkl")
scaler = joblib.load("../models/scaler.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")

def predict_genre(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    genre = label_encoder.inverse_transform(prediction)
    
    return genre[0]

# Example input
sample_song = [120, 0.8, 0.7, -5, 0.05, 0.2, 0.0, 0.1, 0.6]

print("Predicted Genre:", predict_genre(sample_song))