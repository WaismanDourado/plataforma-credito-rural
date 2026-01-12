from pydantic import BaseModel
from typing import Optional

# ---Data Models for User and Authentication---
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    username: str
    password: str

class User(UserBase):
    id: int
    username: str
    is_active: bool = True

    class Config:
        from_attributes = True  # Pydantic v2

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None