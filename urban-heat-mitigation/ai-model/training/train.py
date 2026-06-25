from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "heat_model.pkl"

print("BASE_DIR =", BASE_DIR)
print("DATA_PATH =", DATA_PATH)
print("MODEL_PATH =", MODEL_PATH)
print("Dataset Path:", DATA_PATH)
print("Model Path:", MODEL_PATH)

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# ==================================================
# Features and Target
# ==================================================

X = df[
    [
        "NDVI",
        "NDBI",
        "Latitude",
        "Longitude"
    ]
]

y = df["LST"]

# ==================================================
# Train-Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==================================================
# Train Model
# ==================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ==================================================
# Save Model
# ==================================================

joblib.dump(model, MODEL_PATH)

print("Model Saved Successfully")
print("Saved At:", MODEL_PATH)