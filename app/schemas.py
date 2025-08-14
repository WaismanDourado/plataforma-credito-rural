from pydantic import BaseModel

class PredictionCreate(BaseModel):
    farmer_name: str
    income: float
    crop_type: str

class PredictionResponse(BaseModel):
    id: int
    approval_probability: float
    approved: bool

class Config:
    orm_model = True