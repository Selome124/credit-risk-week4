from pydantic import BaseModel
from typing import List


class CreditRiskRequest(BaseModel):
    features: List[float]


class CreditRiskResponse(BaseModel):
    risk_probability: float
