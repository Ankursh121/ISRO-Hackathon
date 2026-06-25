
import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "heat_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

HOTSPOT_THRESHOLD = 35

def predict_heat(ndvi, ndbi, latitude, longitude):

    input_data = pd.DataFrame(
        [[ndvi, ndbi, latitude, longitude]],
        columns=[
            "NDVI",
            "NDBI",
            "Latitude",
            "Longitude"
        ]
    )

    predicted_lst = model.predict(input_data)[0]

    return {
        "predicted_temperature": round(predicted_lst, 2),
        "hotspot": detect_hotspot(predicted_lst),
        "recommendation": recommendation(ndvi, ndbi)
    }

def recommendation(ndvi, ndbi):

    if ndvi < 0.2 and ndbi > 0.1:
        return "Increase vegetation and install cool roofs"

    elif ndvi < 0.2:
        return "Increase vegetation cover"

    elif ndbi > 0.1:
        return "Promote reflective materials"

    else:
        return "Maintain current urban landscape"

def detect_hotspot(lst):
    return lst > HOTSPOT_THRESHOLD


def recommendation(ndvi, ndbi):

    if ndvi < 0.2 and ndbi > 0.1:
        return "Increase vegetation and install cool roofs"

    elif ndvi < 0.2:
        return "Increase vegetation cover"

    elif ndbi > 0.1:
        return "Promote reflective materials"

    else:
        return "Maintain current urban landscape"

def predict_heat(ndvi, ndbi, latitude, longitude):

    features = pd.DataFrame({
        "NDVI": [ndvi],
        "NDBI": [ndbi],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })

    predicted_lst = model.predict(features)[0]

    return {
    "predicted_temperature": float(round(predicted_lst, 2)),
    "hotspot": bool(detect_hotspot(predicted_lst)),
    "recommendation": recommendation(ndvi, ndbi)
}
if __name__ == "__main__":
    result = predict_heat(
        ndvi=0.15,
        ndbi=0.20,
        latitude=23.25,
        longitude=77.40
    )

    print(result)