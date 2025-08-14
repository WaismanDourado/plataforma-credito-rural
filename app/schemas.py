from pydantic import BaseModel
from typing import List, Optional

# ---Data Models for the Backend (CRUD, etc.)---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class User(BaseModel):
    id: int
    username: str
    email: str

class Config:
    orm_model = True # Enable ORM mapping for Pydantic

# New Model for Credit Forecasting
class CreditApplication(BaseModel):
    income: float
    years_farming: int
    area_hectares: float

class CreditPredictionResult(BaseModel):
    predicted_approval: int
    probability: float