from pydantic import BaseModel
from typing import List, Optional

# ---Data Models for the Backend (CRUD, etc.)---
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    username: str
    password: str

class User(UserBase):
    id: int
    username: str
    is_active = True

class Config:
    orm_model = True # Enable ORM mapping for Pydantic

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None

# New Model for Credit Forecasting
class CreditApplication(BaseModel):
    income: float
    years_farming: int
    area_hectares: float

class CreditPredictionResult(BaseModel):
    predicted_approval: int
    probability: float