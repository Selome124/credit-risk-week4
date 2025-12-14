from fastapi import FastAPI
from src.api.pydantic_models import CreditRequest, CreditResponse


app = FastAPI(title="Credit Risk Model API")

@app.post("/predict", response_model=CreditResponse)
def predict_risk(request: CreditRequest):
    # Dummy logic (replace with real model later)
    risk = 1 if request.loan_amount > request.income else 0
    return CreditResponse(default_risk=risk)
