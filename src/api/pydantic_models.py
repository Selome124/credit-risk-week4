from pydantic import BaseModel

class CreditRequest(BaseModel):
    income: float
    age: int
    loan_amount: float

class CreditResponse(BaseModel):
    default_risk: int
