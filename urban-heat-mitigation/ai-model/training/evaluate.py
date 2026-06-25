from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "heat_model.pkl"

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv(DATA_PATH)

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

# ==================================================
# Load Model
# ==================================================

model = joblib.load(MODEL_PATH)

# ==================================================
# Predictions
# ==================================================

predictions = model.predict(X_test)

# ==================================================
# Evaluation Metrics
# ==================================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Evaluation Results")
print("------------------------")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))