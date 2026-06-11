import os
from pathlib import Path
from dotenv import load_dotenv
from joblib import load

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()

MODEL_PATH = PROJECT_ROOT / "model_dir" / "medbuddy_model.joblib"
model = load(MODEL_PATH)

def predict(input_data: dict):
    # Urutan fitur harus konsisten dengan urutan saat training
    feature_order = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    
    # Ambil values menjadi format list nested [[val1, val2, ...]]
    features = [[input_data[key] for key in feature_order]]

    # Prediksi langsung menggunakan list (scikit-learn support ini)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "prediction": prediction,
        "probability": probability
    }