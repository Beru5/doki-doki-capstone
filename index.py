from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import predict

app = FastAPI(
    title="Doki-Doki REST API",
    version="1.0.0"
)

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "FastAPI on Vercel is secure and running!"}

@app.post("/api/predict")
def predict_heart_disease(input_data: HeartDiseaseInput):
    try:
        data_dict = input_data.model_dump()
        
        result = predict(input_data=data_dict)
        
        return {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "diagnosis": (
                "Heart Disease Detected"
                if result["prediction"] == 1
                else "No Heart Disease Detected"
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
