import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("../data/spotify.csv")

# Select features
features = [
    'tempo', 'energy', 'danceability', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence'
]

X = df[features]
y = df['genre']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  K-Means Clustering

kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("Cluster Distribution:")
print(df['cluster'].value_counts())


#  Supervised Learning
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save Models

os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(le, "../models/label_encoder.pkl")
joblib.dump(kmeans, "../models/kmeans.pkl")

print("\nModels saved successfully!")

#For Visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='cluster', data=df)
plt.title("Cluster Distribution")
plt.show()