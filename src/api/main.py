from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np

from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse

app = FastAPI(title="Credit Risk Model API")

# Load model from MLflow Registry
MODEL_NAME = "credit-risk-model"
MODEL_STAGE = "latest"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/predict", response_model=CreditRiskResponse)
def predict(data: CreditRiskRequest):
    features = np.array(data.features).reshape(1, -1)
    probability = model.predict(features)[0]

    return CreditRiskResponse(risk_probability=float(probability))
