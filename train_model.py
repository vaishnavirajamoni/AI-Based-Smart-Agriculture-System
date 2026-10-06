import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Total records:", len(data))

# Input features
X = data[[
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]]

# Target
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model training completed!")
print("Model Accuracy:", accuracy)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/crop_model.pkl")

print("Model saved successfully!")
print("Location: models/crop_model.pkl")